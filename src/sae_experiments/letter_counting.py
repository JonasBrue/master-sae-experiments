"""
Letter Counting Benchmark Dataset Generator
Based on notebook 19_Triple_Reporting_Letter_Counting_Dynmic_Min_Occ_NO_FILTER
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Extended word database from the notebook
WORD_DATABASE = {
    "common_words": [
        "hello", "world", "python", "apple", "banana", "orange", "computer",
        "science", "letter", "number", "simple", "example", "testing", "machine",
        "learning", "neural", "network", "language", "program", "function",
        "beautiful", "wonderful", "fantastic", "amazing", "excellent", "brilliant",
        "important", "different", "interesting", "difficult", "possible", "necessary",
        "government", "development", "environment", "information", "organization",
        "relationship", "understanding", "communication", "international", "professional",
        "technology", "university", "community", "opportunity", "responsibility",
        "experience", "performance", "management", "department", "improvement",
        "situation", "population", "generation", "education", "application",
        "traditional", "commercial", "industrial", "financial", "political",
        "historical", "cultural", "national", "regional", "personal",
        "general", "special", "original", "additional", "individual",
        "physical", "natural", "social", "local", "global",
        "central", "federal", "medical", "legal", "final",
        "current", "recent", "present", "future", "modern",
        "ancient", "foreign", "famous", "serious", "various",
        "previous", "following", "including", "according", "considering",
        "beginning", "building", "training", "planning", "marketing",
        "shopping", "cooking", "reading", "writing", "speaking",
        "thinking", "working", "playing", "running", "swimming",
        "dancing", "singing", "painting", "drawing", "filming",
    ],
    "double_letters": [
        "committee", "occurrence", "successfully", "millennium", "accommodate",
        "embarrass", "possession", "professional", "aggressive", "assessment",
        "coffee", "balloon", "broccoli", "zucchini", "cappuccino",
        "mozzarella", "spaghetti", "attentive", "attorney",
        "accelerate", "accessible", "accidentally", "accompany",
        "accomplish", "accountant", "addressing", "affection", "aggressive",
        "allegedly", "alleviate", "allocation", "allowance", "announced",
        "appearance", "appliance", "appreciate", "appropriate", "arrangement",
        "assemble", "assessment", "assignment", "associated", "assumption",
        "attachment", "attendance", "attention", "attitude", "attorney",
        "attraction", "battlefield", "beginning", "bellhop", "bittersweet",
        "blizzard", "bookkeeper", "bottleneck", "butterfly", "caffeine",
        "carriage", "caterpillar", "challenge", "channeling", "classical",
        "collecting", "commanding", "committing", "compelling", "connection",
        "controlling", "corruption", "cuttlefish", "dazzling", "depression",
        "disappear", "dissatisfied", "dizziness", "ineffective", "efficient",
    ],
    "triple_letters": [
        "banana", "assassin", "abracadabra", "alabaster", "amalgamate",
        "anagram", "apparatus", "paragraph", "australia", "avatar",
        "barbarian", "basketball", "calabash", "canada", "caravan",
        "cascara", "catamaran", "database", "maharajah", "niagara",
        "panama", "paparazzi", "paranoia", "piranha", "salami",
        "samurai", "savanna", "tarantula", "tartar", "referee",
        "employee", "evergreen", "enterprise", "elsewhere", "envelope",
        "seventeen", "excellence", "experience", "preference", "reference",
        "reverence", "sentence", "sequence", "remember", "whenever",
        "wherever", "everything", "everywhere", "nevertheless", "everybody",
    ],
    "high_frequency": [
        "Mississippi", "Tennessee", "Connecticut", "Massachusetts", "Pennsylvania",
        "Missouri", "Indianapolis", "Minneapolis", "Philadelphia", "California",
        "intelligence", "engineering", "architecture", "electricity", "electrician",
        "statistician", "politician", "technician", "musician", "physician",
        "collaboration", "appreciation", "identification", "transportation", "communication",
        "comprehensive", "representative", "administrative", "investigative", "demonstrative",
    ],
}


def clean_word_database(word_db: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Remove duplicates and invalid words."""
    seen = set()
    cleaned_db = {}
    
    for category, words in word_db.items():
        cleaned_words = []
        for word in words:
            clean_word = word.lower().strip()
            if not clean_word.isalpha():
                continue
            if clean_word in seen:
                continue
            seen.add(clean_word)
            cleaned_words.append(clean_word)
        cleaned_db[category] = cleaned_words
    
    return cleaned_db


def count_letter(word: str, letter: str) -> int:
    """Count occurrences of a letter in a word."""
    return word.lower().count(letter.lower())


def generate_question(word: str, letter: str) -> Tuple[str, str]:
    """Generate a letter counting question."""
    count = count_letter(word, letter)
    
    question = f'How many times does the letter "{letter.upper()}" appear in the word "{word}"?'
    answer = str(count)
    
    return question, answer


def generate_dataset(
    word_database: Dict[str, List[str]],
    questions_per_word: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """Generate letter counting dataset."""
    random.seed(seed)
    
    cleaned_db = clean_word_database(word_database)
    all_words = []
    for category, words in cleaned_db.items():
        all_words.extend(words)
    
    all_words = list(set(all_words))  # Remove duplicates
    random.shuffle(all_words)
    
    questions = []
    
    for word in all_words:
        # Get unique letters in the word
        letters = list(set(word.lower()))
        
        # Sample letters to count
        sample_letters = random.sample(letters, min(questions_per_word, len(letters)))
        
        for letter in sample_letters:
            question_text, answer = generate_question(word, letter)
            
            # Choices are 0-5
            choices = ["0", "1", "2", "3", "4", "5"]
            
            questions.append({
                "question": question_text,
                "answer": answer,
                "word": word,
                "letter": letter,
                "count": int(answer),
                "choices": choices,
            })
    
    return questions


def split_dataset(
    questions: List[Dict],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train/val/test."""
    random.seed(seed)
    
    # Shuffle
    shuffled = questions.copy()
    random.shuffle(shuffled)
    
    # Split
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]
    
    return train, val, test


def save_dataset(questions: List[Dict], path: Path):
    """Save dataset to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"Saved {len(questions)} questions to {path}")


def generate_letter_counting_benchmark(
    output_dir: Path,
    questions_per_word: int = 3,
    seed: int = 42,
) -> Dict[str, Path]:
    """Generate and save letter counting benchmark."""
    
    print("Generating Letter Counting benchmark...")
    print(f"  Questions per word: {questions_per_word}")
    print(f"  Seed: {seed}")
    
    # Generate questions
    questions = generate_dataset(WORD_DATABASE, questions_per_word, seed)
    
    # Split
    train, val, test = split_dataset(questions, seed=seed)
    
    print(f"  Total questions: {len(questions)}")
    print(f"  Train: {len(train)}")
    print(f"  Val: {len(val)}")
    print(f"  Test: {len(test)}")
    
    # Save
    paths = {}
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = output_dir / f"letter_counting_{split_name}.json"
        save_dataset(split_data, path)
        paths[split_name] = path
    
    return paths


if __name__ == "__main__":
    output_dir = Path("/workspace/letter_counting")
    paths = generate_letter_counting_benchmark(output_dir)
    print("\nGenerated files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
