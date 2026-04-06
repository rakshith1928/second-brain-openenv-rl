"""
Seed data for the Second Brain environment.
Fixed seed ensures reproducible scores across all runs.
"""
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Task 1 — 10 notes to categorize
# ---------------------------------------------------------------------------
TASK1_NOTES: List[Dict[str, Any]] = [
    {
        "id": "n001",
        "text": "Follow up with Priya about the Q3 budget report by Friday.",
        "correct_category": "action_item",
    },
    {
        "id": "n002",
        "text": "Interesting article: how sleep affects memory consolidation. Save for later.",
        "correct_category": "reference",
    },
    {
        "id": "n003",
        "text": "Team standup notes: backend blocked on auth service, frontend ready to demo.",
        "correct_category": "work",
    },
    {
        "id": "n004",
        "text": "Mom's birthday is on the 18th. Order cake from the bakery near home.",
        "correct_category": "personal",
    },
    {
        "id": "n005",
        "text": "Research paper on vector databases — useful for the recommendation engine project.",
        "correct_category": "reference",
    },
    {
        "id": "n006",
        "text": "Book dentist appointment before end of month.",
        "correct_category": "action_item",
    },
    {
        "id": "n007",
        "text": "1:1 with manager: discussed promotion timeline, need to finish tech spec first.",
        "correct_category": "work",
    },
    {
        "id": "n008",
        "text": "Watched a great documentary about the ocean. Want to go snorkeling someday.",
        "correct_category": "personal",
    },
    {
        "id": "n009",
        "text": "Python typing cheatsheet — quick reference for generics and protocols.",
        "correct_category": "reference",
    },
    {
        "id": "n010",
        "text": "Send invoice #1042 to client by Thursday.",
        "correct_category": "action_item",
    },
]

# ---------------------------------------------------------------------------
# Task 2 — 30-note knowledge base + 5 retrieval questions
# ---------------------------------------------------------------------------
TASK2_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {"id": "k001", "text": "Meeting with John on March 5: project deadline pushed to April 15 due to design review.", "topic": "work"},
    {"id": "k002", "text": "Gym routine: Mon/Wed/Fri — upper body. Tue/Thu — cardio. Rest on weekends.", "topic": "personal"},
    {"id": "k003", "text": "Book recommendation from Ananya: 'Deep Work' by Cal Newport.", "topic": "reference"},
    {"id": "k004", "text": "API rate limit for the weather service is 1000 calls/day on free tier.", "topic": "work"},
    {"id": "k005", "text": "Grocery list: oats, almond milk, spinach, Greek yogurt, eggs.", "topic": "personal"},
    {"id": "k006", "text": "React useEffect cleanup: return a function to cancel subscriptions and avoid memory leaks.", "topic": "reference"},
    {"id": "k007", "text": "Call with Priya: she needs the analytics dashboard by end of sprint (March 20).", "topic": "work"},
    {"id": "k008", "text": "Interesting talk by Andrew Ng on data-centric AI — focus on data quality over model tuning.", "topic": "reference"},
    {"id": "k009", "text": "Visa appointment booked for June 10 at 9 AM. Bring all original documents.", "topic": "personal"},
    {"id": "k010", "text": "Tech spec for recommendation engine: use collaborative filtering + content-based hybrid.", "topic": "work"},
    {"id": "k011", "text": "Password manager: use 1Password for all work accounts. Enable 2FA everywhere.", "topic": "reference"},
    {"id": "k012", "text": "Flight to Bangalore: April 25, 6:15 AM. Check-in online 24 hours before.", "topic": "personal"},
    {"id": "k013", "text": "Sprint retrospective: team agreed to reduce WIP limit to 3 tasks per person.", "topic": "work"},
    {"id": "k014", "text": "How to center a div in CSS: use flexbox with justify-content and align-items set to center.", "topic": "reference"},
    {"id": "k015", "text": "Dad's medical checkup results came back normal. Next appointment in 6 months.", "topic": "personal"},
    {"id": "k016", "text": "Vendor contract renewal due on May 1. Legal team needs 2 weeks notice.", "topic": "work"},
    {"id": "k017", "text": "Python tip: use dataclasses for simple data containers, Pydantic when you need validation.", "topic": "reference"},
    {"id": "k018", "text": "Movie night idea: Dune Part 2, Everything Everywhere, Past Lives.", "topic": "personal"},
    {"id": "k019", "text": "Architecture decision: move from monolith to microservices in Q3. Start with auth service.", "topic": "work"},
    {"id": "k020", "text": "Git workflow: feature branches off main, squash commits before merging, tag releases.", "topic": "reference"},
    {"id": "k021", "text": "Meditation app: Headspace — doing 10 min daily, helps with focus at work.", "topic": "personal"},
    {"id": "k022", "text": "Budget for Q2: engineering headcount approved for 2 new hires.", "topic": "work"},
    {"id": "k023", "text": "Docker tip: use multi-stage builds to reduce final image size significantly.", "topic": "reference"},
    {"id": "k024", "text": "Weekend plan: visit the botanical garden on Saturday, brunch with friends Sunday.", "topic": "personal"},
    {"id": "k025", "text": "OKR for Q2: increase API response time by 30%, reduce error rate below 0.1%.", "topic": "work"},
    {"id": "k026", "text": "Learning resource: fast.ai course for practical deep learning — free and hands-on.", "topic": "reference"},
    {"id": "k027", "text": "Apartment lease renewal in August. Start looking at options in June.", "topic": "personal"},
    {"id": "k028", "text": "Code review checklist: check for error handling, test coverage, naming consistency.", "topic": "work"},
    {"id": "k029", "text": "Pomodoro technique: 25 min focus + 5 min break. After 4 rounds take a longer break.", "topic": "reference"},
    {"id": "k030", "text": "Sister's wedding anniversary gift: book a dinner reservation at that Italian restaurant.", "topic": "personal"},
]

TASK2_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "q001",
        "question": "What is the deadline for the project that John mentioned?",
        "correct_note_id": "k001",
        "keywords": ["john", "deadline", "april", "project"],
    },
    {
        "id": "q002",
        "question": "What is the API rate limit for the weather service?",
        "correct_note_id": "k004",
        "keywords": ["api", "rate limit", "weather", "calls"],
    },
    {
        "id": "q003",
        "question": "When is the visa appointment and what should I bring?",
        "correct_note_id": "k009",
        "keywords": ["visa", "appointment", "documents", "june"],
    },
    {
        "id": "q004",
        "question": "What architecture approach was decided for Q3?",
        "correct_note_id": "k019",
        "keywords": ["architecture", "microservices", "monolith", "auth"],
    },
    {
        "id": "q005",
        "question": "What is the Python tip about when to use Pydantic vs dataclasses?",
        "correct_note_id": "k017",
        "keywords": ["python", "pydantic", "dataclass", "validation"],
    },
]

# ---------------------------------------------------------------------------
# Task 3 — 50-note knowledge base + 3 synthesis questions
# ---------------------------------------------------------------------------
# Extends Task 2 KB with 20 more notes
TASK3_EXTRA_NOTES: List[Dict[str, Any]] = [
    {"id": "k031", "text": "Feeling overwhelmed with workload this week. Too many meetings, no deep work time.", "topic": "work_stress"},
    {"id": "k032", "text": "Skipped gym 3 days this week due to late meetings. Energy levels low.", "topic": "health"},
    {"id": "k033", "text": "Had a great 1:1 with mentor: she suggested blocking 2 hours daily for focused coding.", "topic": "work"},
    {"id": "k034", "text": "Headache for 2 days. Might be from screen time. Need to take more breaks.", "topic": "health"},
    {"id": "k035", "text": "Read article: chronic stress reduces productivity by up to 40% over time.", "topic": "reference"},
    {"id": "k036", "text": "Sprint planning took 3 hours — too long. Propose async planning next time.", "topic": "work_stress"},
    {"id": "k037", "text": "Back pain from sitting too long. Started doing 5 min stretches every hour.", "topic": "health"},
    {"id": "k038", "text": "Team morale low after missed deadline. Need to celebrate small wins more.", "topic": "work_stress"},
    {"id": "k039", "text": "Sleep only 5 hours last 3 nights. Affecting concentration during code reviews.", "topic": "health"},
    {"id": "k040", "text": "Cancelled weekend plans because of urgent production bug fix. Feeling burnt out.", "topic": "work_stress"},
    {"id": "k041", "text": "Doctor visit: advised 7-8 hours sleep, regular exercise, reduce caffeine.", "topic": "health"},
    {"id": "k042", "text": "Tried async standup instead of meeting — saved 45 min and felt less rushed.", "topic": "work"},
    {"id": "k043", "text": "Coffee intake: 4 cups per day this week. Feel jittery and anxious.", "topic": "health"},
    {"id": "k044", "text": "Project scope keeps expanding — need to push back on new feature requests.", "topic": "work_stress"},
    {"id": "k045", "text": "Went for a 30 min walk during lunch — felt much more focused in the afternoon.", "topic": "health"},
    {"id": "k046", "text": "Boundary setting: stopped checking Slack after 8 PM. Sleep improved immediately.", "topic": "work_stress"},
    {"id": "k047", "text": "Research: exercise improves cognitive function and reduces stress hormones.", "topic": "reference"},
    {"id": "k048", "text": "Team agreed: no meetings on Fridays. Using it for deep work and documentation.", "topic": "work"},
    {"id": "k049", "text": "Mood tracker: consistently lower energy on days with 4+ meetings.", "topic": "health"},
    {"id": "k050", "text": "Note to self: when stressed, the first thing I skip is exercise. That makes it worse.", "topic": "health"},
]

TASK3_KNOWLEDGE_BASE = TASK2_KNOWLEDGE_BASE + TASK3_EXTRA_NOTES

TASK3_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "sq001",
        "question": "What patterns connect my work stress notes to my health notes? What should I do?",
        "relevant_note_ids": ["k031", "k032", "k034", "k036", "k037", "k039", "k040", "k043", "k046", "k049", "k050"],
        "key_themes": ["stress", "health", "exercise", "sleep", "meetings"],
        "expected_insight": "High meeting load → skipped exercise → poor sleep → low energy → more stress. Breaking the cycle requires protecting exercise time and setting meeting boundaries.",
    },
    {
        "id": "sq002",
        "question": "What technical decisions have been made for Q2/Q3 and are they consistent?",
        "relevant_note_ids": ["k010", "k013", "k019", "k022", "k025", "k028", "k042", "k048"],
        "key_themes": ["architecture", "microservices", "okr", "async", "meetings"],
        "expected_insight": "Q3 moves to microservices starting with auth, Q2 focuses on performance OKRs. Async work patterns are being adopted (async standup, no-meeting Fridays). Consistent theme: reducing synchronous overhead.",
    },
    {
        "id": "sq003",
        "question": "What lifestyle changes have I tried and what actually worked?",
        "relevant_note_ids": ["k021", "k037", "k041", "k042", "k045", "k046", "k048", "k050"],
        "key_themes": ["meditation", "exercise", "boundaries", "sleep", "productivity"],
        "expected_insight": "Boundary setting (no Slack after 8pm) and exercise (lunch walks) had immediate positive effects. Meditation helps focus. The challenge is maintaining these when work stress peaks.",
    },
]

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------
VALID_CATEGORIES = ["work", "personal", "reference", "action_item"]

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def keyword_overlap_score(query: str, note_text: str) -> float:
    """
    Simple keyword overlap score between a query and a note.
    Returns a float in [0, 1].
    """
    query_words = set(query.lower().split())
    note_words = set(note_text.lower().split())
    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "in", "on", "at", "to", "for",
                  "of", "and", "or", "but", "with", "my", "i", "what",
                  "how", "when", "where", "which", "that", "this", "it"}
    query_words -= stop_words
    note_words -= stop_words
    if not query_words:
        return 0.0
    overlap = query_words & note_words
    return len(overlap) / len(query_words)