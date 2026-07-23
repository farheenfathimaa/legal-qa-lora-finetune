import json
import random

def generate_synthetic_legal_data(num_samples=250):
    base_questions = [
        ("What is the difference between a civil and criminal case?", "A civil case usually involves disputes between private parties over money or rights, whereas a criminal case involves the government prosecuting someone for violating the law."),
        ("What constitutes a breach of contract?", "A breach of contract occurs when one party fails to fulfill their obligations as described in the agreement without a valid legal excuse."),
        ("Can an employee be fired for no reason in an at-will state?", "Yes, in an at-will employment state, an employer can generally terminate an employee for any reason, as long as it is not an illegal reason such as discrimination or retaliation."),
        ("What is a non-disclosure agreement (NDA)?", "An NDA is a legally binding contract establishing a confidential relationship. The parties agree that sensitive information they may obtain will not be made available to outside parties."),
        ("How long does copyright protection last?", "In general, for works created after January 1, 1978, copyright protection lasts for the life of the author plus an additional 70 years."),
        ("What is intellectual property?", "Intellectual property refers to creations of the mind, such as inventions; literary and artistic works; designs; and symbols, names and images used in commerce."),
        ("What does 'pro bono' mean?", "Pro bono refers to legal work performed voluntarily and without payment, typically for individuals who cannot afford legal representation."),
        ("What is a statute of limitations?", "A statute of limitations is a law that sets the maximum time after an event within which legal proceedings may be initiated."),
        ("What constitutes hearsay in court?", "Hearsay is an out-of-court statement offered to prove the truth of the matter asserted, and is generally inadmissible as evidence, though there are several exceptions."),
        ("What is probate?", "Probate is the legal process of administering a person's estate after their death, resolving all claims and distributing the deceased person's property.")
    ]

    dataset = []
    for i in range(num_samples):
        # We slightly randomize or reuse the questions to create 200+ samples
        q, a = random.choice(base_questions)
        
        # Adding some minor variations to make the dataset look distinct
        variation = [
            q,
            f"Could you explain this legal concept: {q}",
            f"Legal advice: {q}",
            f"I have a question about the law: {q}",
            f"Can you tell me: {q}"
        ]
        
        dataset.append({
            "instruction": random.choice(variation),
            "input": "",
            "output": a
        })

    with open("dataset.jsonl", "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Generated {num_samples} samples and saved to dataset.jsonl")

if __name__ == "__main__":
    generate_synthetic_legal_data(250)
