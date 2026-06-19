"""Central configuration constants for the v2 pipeline.

This module mirrors the S0/S1 build contract in ``docs/planning_v2.md``.
YAML files can still hold experiment-specific settings, but core invariants
live here so every runner can share the same defaults.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

LLM_MODEL = "Qwen/Qwen3.5-4B"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"  # planning_v2.md: "BAAI/bge-reranker-v2-m3 or Qwen/Qwen3-Reranker-0.6B"

GPU_MEM_UTIL = 0.85
MARGIN_LOW = 0.15
SC_N = 5
SC_TEMP = 0.6
RERANK_MIN = 0.5

FORCE_RETRIEVE_DOMAINS = {
    "vn_law",
    "vn_decree",
    "vn_admin",
    "local_facts",
}

TOK = {
    "READING": 512,
    "STEM": 3072,
    "KNOWLEDGE": 256,
    "SAFETY": 128,
}

FALLBACK = "A"
MAX_CHOICES = 26

SUBJECT_META: dict[str, dict[str, str]] = {
    "01": {"name": "Elementary Mathematics",                                     "category": "STEM"},
    "02": {"name": "Elementary Science",                                          "category": "STEM"},
    "03": {"name": "Middle School Biology",                                       "category": "STEM"},
    "04": {"name": "Middle School Chemistry",                                     "category": "STEM"},
    "05": {"name": "Middle School Mathematics",                                   "category": "STEM"},
    "06": {"name": "Middle School Physics",                                       "category": "STEM"},
    "07": {"name": "High School Biology",                                         "category": "STEM"},
    "08": {"name": "High School Chemistry",                                       "category": "STEM"},
    "09": {"name": "High School Mathematics",                                     "category": "STEM"},
    "10": {"name": "High School Physics",                                         "category": "STEM"},
    "11": {"name": "Applied Informatics",                                         "category": "STEM"},
    "12": {"name": "Computer Architecture",                                       "category": "STEM"},
    "13": {"name": "Computer Network",                                            "category": "STEM"},
    "14": {"name": "Discrete Mathematics",                                        "category": "STEM"},
    "15": {"name": "Electrical Engineering",                                      "category": "STEM"},
    "16": {"name": "Introduction to Chemistry",                                   "category": "STEM"},
    "17": {"name": "Introduction to Physics",                                     "category": "STEM"},
    "18": {"name": "Introduction to Programming",                                 "category": "STEM"},
    "19": {"name": "Metrology Engineer",                                          "category": "STEM"},
    "20": {"name": "Operating System",                                            "category": "STEM"},
    "21": {"name": "Statistics and Probability",                                  "category": "STEM"},
    "22": {"name": "Middle School Civil Education",                               "category": "Social Science"},
    "23": {"name": "Middle School Geography",                                     "category": "Social Science"},
    "24": {"name": "High School Civil Education",                                 "category": "Social Science"},
    "25": {"name": "High School Geography",                                       "category": "Social Science"},
    "26": {"name": "Business Administration",                                     "category": "Social Science"},
    "27": {"name": "Ho Chi Minh Ideology",                                        "category": "Social Science"},
    "28": {"name": "Macroeconomics",                                              "category": "Social Science"},
    "29": {"name": "Microeconomics",                                              "category": "Social Science"},
    "30": {"name": "Principles of Marxism and Leninism",                          "category": "Social Science"},
    "31": {"name": "Sociology",                                                   "category": "Social Science"},
    "32": {"name": "Elementary History",                                          "category": "Humanity"},
    "33": {"name": "Middle School History",                                       "category": "Humanity"},
    "34": {"name": "Middle School Literature",                                    "category": "Humanity"},
    "35": {"name": "High School History",                                         "category": "Humanity"},
    "36": {"name": "High School Literature",                                      "category": "Humanity"},
    "37": {"name": "Administrative Law",                                          "category": "Humanity"},
    "38": {"name": "Business Law",                                                "category": "Humanity"},
    "39": {"name": "Civil Law",                                                   "category": "Humanity"},
    "40": {"name": "Criminal Law",                                                "category": "Humanity"},
    "41": {"name": "Economic Law",                                                "category": "Humanity"},
    "42": {"name": "Education Law",                                               "category": "Humanity"},
    "43": {"name": "History of World Civilization",                               "category": "Humanity"},
    "44": {"name": "Ideological and Moral Cultivation",                           "category": "Humanity"},
    "45": {"name": "Introduction to Laws",                                        "category": "Humanity"},
    "46": {"name": "Introduction to Vietnam Culture",                             "category": "Humanity"},
    "47": {"name": "Logic",                                                       "category": "Humanity"},
    "48": {"name": "Revolutionary Policy of the Vietnamese Communist Party",      "category": "Humanity"},
    "49": {"name": "Vietnamese Language and Literature",                          "category": "Humanity"},
    "50": {"name": "Accountant",                                                  "category": "Other"},
    "51": {"name": "Clinical Pharmacology",                                       "category": "Other"},
    "52": {"name": "Environmental Engineering",                                   "category": "Other"},
    "53": {"name": "Internal Basic Medicine",                                     "category": "Other"},
    "54": {"name": "Preschool Pedagogy",                                          "category": "Other"},
    "55": {"name": "Tax Accountant",                                              "category": "Other"},
    "56": {"name": "Tax Civil Servant",                                           "category": "Other"},
    "57": {"name": "Civil Servant",                                               "category": "Other"},
    "58": {"name": "Driving License Certificate",                                 "category": "Other"},
}
