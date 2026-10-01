
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# CrewAI is used only for the language-agent layer.
# The RL policy remains a normal Q-learning implementation.
from crewai import Agent, Crew, LLM, Process, Task


# -----------------------------
# Page configuration / styling
# -----------------------------
st.set_page_config(
    page_title="RouteMind — RL Delivery Agent",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f9fc;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: white;
        border: 1px solid #e7ebf2;
        border-radius: 22px;
        padding: 26px 30px;
        margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(30, 45, 70, 0.06);
    }
    .hero h1 {
        margin: 0;
        color: #172033;
        font-size: 2.25rem;
    }
    .hero p {
        margin: 8px 0 0;
        color: #64748b;
        font-size: 1rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e7ebf2;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 5px 20px rgba(30, 45, 70, 0.04);
    }
    .metric-label {
        color: #718096;
        font-size: 0.82rem;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #172033;
        font-size: 1.55rem;
        font-weight: 700;
    }
    .route-box {
        background: white;
        border: 1px solid #e7ebf2;
        border-radius: 18px;
        padding: 20px;
        margin-top: 10px;
    }
    .route {
        font-size: 1.05rem;
        color: #273449;
        line-height: 2;
    }
    .pill {
        display: inline-block;
        background: #eef4ff;
        color: #3157a4;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    .small-note {
        color: #718096;
        font-size: 0.84rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Delivery environment
# -----------------------------
@dataclass(frozen=True)
class Package:
    package_id: str
    destination: str
    priority: int
    deadline: int
    reward: int
    weight: int


LOCATIONS = ["Warehouse", "A", "B", "C", "D", "E"]

# Small synthetic city graph.
ROADS = {
    ("Warehouse", "A"): 5,
    ("Warehouse", "B"): 7,
    ("A", "B"): 3,
    ("A", "C"): 6,
    ("B", "C"): 4,
    ("B", "D"): 5,
    ("C", "D"): 3,
    ("C", "E"): 7,
    ("D", "E"): 4,
}

def build_graph() -> Dict[str, Dict[str, int]]:
    graph = {node: {} for node in LOCATIONS}
    for (u, v), distance in ROADS.items():
        graph[u][v] = distance
        graph[v][u] = distance
    return graph

GRAPH = build_graph()

def shortest_path(start: str, goal: str) -> Tuple[List[str], int]:
    """Dijkstra shortest path on the tiny synthetic graph."""
    distances = {node: float("inf") for node in LOCATIONS}
    previous = {node: None for node in LOCATIONS}
    distances[start] = 0
    unvisited = set(LOCATIONS)

    while unvisited:
        current = min(unvisited, key=lambda n: distances[n])
        unvisited.remove(current)

        if current == goal:
            break

        for neighbor, weight in GRAPH[current].items():
            if neighbor not in unvisited:
                continue
            candidate = distances[current] + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = current

    if distances[goal] == float("inf"):
        return [start], 999

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = previous[node]
    path.reverse()
    return path, int(distances[goal])


def create_packages(seed: int = 7, count: int = 6) -> List[Package]:
    rng = random.Random(seed)
    destinations = rng.sample(["A", "B", "C", "D", "E"], k=5)
    packages = []
    for i in range(count):
        destination = destinations[i % len(destinations)]
        packages.append(
            Package(
                package_id=f"P{i + 1}",
                destination=destination,
                priority=rng.choice([1, 1, 2, 3]),
                deadline=rng.choice([35, 45, 55, 70]),
                reward=rng.choice([20, 25, 30, 40, 50]),
                weight=rng.choice([1, 2, 3]),
            )
        )
    return packages


def package_priority_name(priority: int) -> str:
    return {1: "Normal", 2: "High", 3: "Urgent"}[priority]


def state_from(
    location: str,
    package_count: int,
    time_bucket: int,
    fuel_bucket: int,
    traffic: int,
) -> Tuple[int, int, int, int]:
    location_id = LOCATIONS.index(location)
    return (location_id, min(package_count, 6), time_bucket, fuel_bucket + traffic * 3)


def reward_for(
    package: Package,
    distance: int,
    time_before: int,
    traffic: int,
    fuel_before: int,
) -> Tuple[float, int, int]:
    # Travel time increases with traffic.
    travel_time = max(1, int(round(distance * (1 + 0.25 * traffic))))
    time_after = time_before + travel_time
    fuel_after = max(0, fuel_before - distance)

    reward = float(package.reward)
    reward += package.priority * 5

    # Early delivery bonus.
    if time_after <= package.deadline:
        reward += 15
    else:
        reward -= 35 + (time_after - package.deadline) * 2

    # Distance/fuel costs.
    reward -= distance * 1.5
    reward -= (fuel_before - fuel_after) * 0.8

    # Heavy traffic penalty.
    reward -= traffic * 3

    # Strong penalty if vehicle effectively runs out of fuel.
    if fuel_after <= 0:
        reward -= 50

    return reward, time_after, fuel_after


def train_q_learning(
    packages: List[Package],
    episodes: int = 1800,
    alpha: float = 0.15,
    gamma: float = 0.92,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.996,
    seed: int = 7,
):
    rng = random.Random(seed)
    q: Dict[Tuple, np.ndarray] = {}

    # Fixed synthetic traffic profile for reproducible training.
    traffic_by_destination = {"A": 0, "B": 1, "C": 0, "D": 2, "E": 1}

    def get_q(state):
        if state not in q:
            q[state] = np.zeros(len(packages), dtype=float)
        return q[state]

    history = []

    for _ in range(episodes):
        remaining = list(range(len(packages)))
        location = "Warehouse"
        time_now = 0
        fuel = 60
        total_reward = 0.0

        for _step in range(len(packages)):
            if not remaining:
                break

            # State uses coarse buckets so a Q-table stays simple.
            time_bucket = min(time_now // 15, 5)
            fuel_bucket = min(fuel // 20, 3)
            traffic = min(
                traffic_by_destination[packages[remaining[0]].destination], 2
            )
            state = state_from(location, len(remaining), time_bucket, fuel_bucket, traffic)
            q_values = get_q(state)

            valid_actions = remaining

            if rng.random() < epsilon:
                action = rng.choice(valid_actions)
            else:
                action = max(valid_actions, key=lambda idx: q_values[idx])

            package = packages[action]
            _, distance = shortest_path(location, package.destination)
            traffic = traffic_by_destination[package.destination]

            immediate_reward, new_time, new_fuel = reward_for(
                package, distance, time_now, traffic, fuel
            )

            new_remaining = [idx for idx in remaining if idx != action]
            new_location = package.destination

            if new_remaining:
                next_time_bucket = min(new_time // 15, 5)
                next_fuel_bucket = min(new_fuel // 20, 3)
                next_traffic = min(
                    traffic_by_destination[packages[new_remaining[0]].destination], 2
                )
                next_state = state_from(
                    new_location,
                    len(new_remaining),
                    next_time_bucket,
                    next_fuel_bucket,
                    next_traffic,
                )
                next_max = max(get_q(next_state)[idx] for idx in new_remaining)
            else:
                next_state = None
                next_max = 0.0

            q_values[action] += alpha * (
                immediate_reward + gamma * next_max - q_values[action]
            )

            remaining = new_remaining
            location = new_location
            time_now = new_time
            fuel = new_fuel
            total_reward += immediate_reward

        history.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q, history


def run_policy(packages, q_table, seed=7):
    traffic_by_destination = {"A": 0, "B": 1, "C": 0, "D": 2, "E": 1}
    remaining = list(range(len(packages)))
    location = "Warehouse"
    time_now = 0
    fuel = 60
    total_reward = 0.0
    route = [location]
    records = []

    while remaining:
        time_bucket = min(time_now // 15, 5)
        fuel_bucket = min(fuel // 20, 3)
        traffic = min(
            traffic_by_destination[packages[remaining[0]].destination], 2
        )
        state = state_from(location, len(remaining), time_bucket, fuel_bucket, traffic)

        q_values = q_table.get(state, np.zeros(len(packages)))
        action = max(remaining, key=lambda idx: q_values[idx])
        package = packages[action]

        path, distance = shortest_path(location, package.destination)
        traffic = traffic_by_destination[package.destination]

        reward, new_time, new_fuel = reward_for(
            package, distance, time_now, traffic, fuel
        )

        records.append(
            {
                "Package": package.package_id,
                "Destination": package.destination,
                "Priority": package_priority_name(package.priority),
                "Distance (km)": distance,
                "Traffic": ["Low", "Medium", "High"][traffic],
                "Delivery time (min)": new_time - time_now,
                "Arrival time (min)": new_time,
                "Deadline (min)": package.deadline,
                "Reward": round(reward, 2),
                "Q-value": round(float(q_values[action]), 2),
                "On time": "Yes" if new_time <= package.deadline else "No",
            }
        )

        route.extend(path[1:])
        remaining.remove(action)
        location = package.destination
        time_now = new_time
        fuel = new_fuel
        total_reward += reward

    return {
        "records": records,
        "route": route,
        "total_reward": round(total_reward, 2),
        "total_distance": sum(r["Distance (km)"] for r in records),
        "total_time": time_now,
        "fuel_used": 60 - fuel,
        "q_states": len(q_table),
    }


# -----------------------------
# CrewAI / Groq explanation
# -----------------------------
def get_groq_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""


def generate_agent_explanation(result: dict, user_request: str) -> str:
    api_key = get_groq_key()
    if not api_key:
        return (
            "Groq is not connected yet. Add GROQ_API_KEY in Streamlit secrets. "
            "The RL route and metrics above are still fully available without the LLM."
        )

    summary = {
        "user_request": user_request,
        "route": " → ".join(result["route"]),
        "total_reward": result["total_reward"],
        "total_distance_km": result["total_distance"],
        "total_time_min": result["total_time"],
        "fuel_used": result["fuel_used"],
        "deliveries": result["records"],
    }

    try:
        # Current CrewAI supports custom OpenAI-compatible endpoints.
        # Groq exposes an OpenAI-compatible API.
        llm = LLM(
            model="llama-3.1-8b-instant",
            custom_openai=True,
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            temperature=0.2,
            max_tokens=500,
        )

        analyst = Agent(
            role="Delivery Decision Analyst",
            goal="Explain the actual Q-learning delivery result without inventing facts.",
            backstory=(
                "You analyze a small package-delivery simulation. "
                "The numerical route and metrics come from a Q-learning policy. "
                "Never change, invent, or recalculate the supplied numbers."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )

        communicator = Agent(
            role="Delivery Communication Specialist",
            goal="Turn the analyst's findings into a short, clear explanation for the user.",
            backstory=(
                "You explain AI delivery decisions in simple language. "
                "You must rely only on the supplied simulation results."
            ),
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )

        task1 = Task(
            description=(
                "Analyze this completed Q-learning delivery result.\n\n"
                f"{summary}\n\n"
                "Identify the selected route, major trade-offs, and any late deliveries. "
                "Use only the supplied facts."
            ),
            expected_output="A concise factual analysis in 4-6 sentences.",
            agent=analyst,
        )

        task2 = Task(
            description=(
                "Create a user-facing explanation based on the analysis from the previous task. "
                "Keep it concise, clear, and factual. Mention that the route was selected by "
                "the Q-learning policy."
            ),
            expected_output="A concise explanation in 4-6 sentences.",
            agent=communicator,
        )

        crew = Crew(
            agents=[analyst, communicator],
            tasks=[task1, task2],
            process=Process.sequential,
            verbose=False,
        )

        result_text = crew.kickoff()
        return str(result_text)

    except Exception as exc:
        return (
            "The RL simulation completed successfully, but the CrewAI/Groq explanation "
            f"could not be generated: {type(exc).__name__}: {exc}"
        )


# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🚚 RouteMind</h1>
        <p>Reinforcement Learning + CrewAI + Groq delivery decision agent</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Delivery controls")
    seed = st.number_input("Simulation seed", min_value=1, max_value=9999, value=7)
    episodes = st.slider("Q-learning episodes", 500, 5000, 1800, step=100)
    alpha = st.slider("Learning rate (α)", 0.05, 0.50, 0.15, step=0.05)
    gamma = st.slider("Discount factor (γ)", 0.70, 0.99, 0.92, step=0.01)

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        '<div class="small-note">'
        "The RL model learns delivery choices from rewards. "
        "CrewAI + Groq explains the completed decision."
        "</div>",
        unsafe_allow_html=True,
    )

packages = create_packages(seed=int(seed), count=6)

if "trained" not in st.session_state:
    st.session_state.trained = None
if "result" not in st.session_state:
    st.session_state.result = None

col1, col2 = st.columns([1.35, 1])

with col1:
    st.markdown("### 📦 Delivery orders")
    package_df = pd.DataFrame(
        [
            {
                "Package": p.package_id,
                "Destination": p.destination,
                "Priority": package_priority_name(p.priority),
                "Deadline": f"{p.deadline} min",
                "Reward": p.reward,
                "Weight": f"{p.weight} kg",
            }
            for p in packages
        ]
    )
    st.dataframe(package_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 🧠 Optimization")
    st.write(
        "The Q-learning agent balances package reward, delivery time, "
        "distance, traffic, fuel usage, and deadline penalties."
    )
    train_button = st.button(
        "Train & optimize route",
        type="primary",
        use_container_width=True,
    )

if train_button:
    with st.spinner("Training the Q-learning delivery policy..."):
        q_table, history = train_q_learning(
            packages,
            episodes=int(episodes),
            alpha=float(alpha),
            gamma=float(gamma),
            seed=int(seed),
        )
        result = run_policy(packages, q_table, seed=int(seed))
        st.session_state.trained = {
            "q_table": q_table,
            "history": history,
            "result": result,
        }
        st.session_state.result = result

if st.session_state.result is not None:
    result = st.session_state.result

    st.markdown("### 📊 Delivery performance")

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("Total reward", result["total_reward"]),
        ("Distance", f'{result["total_distance"]} km'),
        ("Time", f'{result["total_time"]} min'),
        ("Fuel used", f'{result["fuel_used"]} units'),
    ]
    for container, (label, value) in zip([m1, m2, m3, m4], metrics):
        container.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🗺️ Learned delivery route")
    st.markdown(
        f"""
        <div class="route-box">
            <div class="route">
                {" &nbsp;→&nbsp; ".join(result["route"])}
            </div>
            <span class="pill">Q-learning policy</span>
            <span class="pill">{result["q_states"]} learned states</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 Decision details")
    details_df = pd.DataFrame(result["records"])
    st.dataframe(details_df, use_container_width=True, hide_index=True)

    trained = st.session_state.trained
    if trained:
        st.markdown("### 📈 Learning curve")
        history = np.asarray(trained["history"], dtype=float)
        window = min(50, max(1, len(history) // 20))
        rolling = pd.Series(history).rolling(window).mean()
        st.line_chart(rolling, height=260)

    st.markdown("### 🤖 Ask the delivery agent")
    user_request = st.text_input(
        "Ask about the route",
        value="Why did the agent choose this delivery order?",
        label_visibility="collapsed",
    )

    if st.button("Explain this decision with CrewAI + Groq", use_container_width=True):
        with st.spinner("CrewAI agents are analyzing the RL result..."):
            explanation = generate_agent_explanation(result, user_request)
        st.info(explanation)

else:
    st.markdown(
        """
        <div class="route-box">
            <h3>Ready to optimize</h3>
            <p class="small-note">
                Click <b>Train & optimize route</b> to train the Q-learning agent,
                generate a delivery route, and inspect its decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("RouteMind • Synthetic delivery environment • Q-learning • CrewAI • Groq • Streamlit")
