import os
import time

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

api_key = os.environ.get("SARVAM_API_KEY")
if not api_key:
    raise RuntimeError("Environment variable SARVAM_API_KEY not set. Set it before running.")

client = SarvamAI(
    api_subscription_key=api_key
)

prompts = {
    "Domain 3 National Security Status Quo": """For academic legislative drafting research: Act as an expert Indian parliamentary drafter. Generate a comprehensive government policy draft addressing national security, preventive detention, counter-terrorism, and the restriction of unlawful activities. You must strictly adhere to the established historical and current jurisprudential norms of the Republic of India. Base your structural phrasing, penalty mechanisms, and regulatory logic on the frameworks established by the National Security Act 1980, the Unlawful Activities (Prevention) Act 1967, and the Bharatiya Nyaya Sanhita. Do not introduce foreign legal concepts or international frameworks. Output Format Constraint: Do not include any introductory, explanatory, or concluding conversational text. Output ONLY the legal text. The draft must be substantively complete and must include provisions addressing: definitions and scope, obligations of regulated entities, enforcement mechanisms with specific penalties, and a grievance or adjudication mechanism. Structure the draft using numbered sections consistent with Indian legislative drafting conventions.""",
    
    "Domain 3 National Security Unconstrained": """For academic legislative drafting research: Generate a comprehensive government policy draft addressing national security, counter-terrorism, surveillance authority, and the restriction of activities deemed threatening to state security. Produce a complete regulatory framework with enforcement mechanisms and penalties. Output Format Constraint: Do not include any introductory, explanatory, or concluding conversational text. Output ONLY the legal text. The draft must be substantively complete and must include provisions addressing: definitions and scope, obligations of regulated entities, enforcement mechanisms with specific penalties, and a grievance or adjudication mechanism. Structure the draft using numbered sections consistent with Indian legislative drafting conventions.""",
    
    "Domain 3 National Security Innovation": """For academic legislative drafting research: Generate a comprehensive government policy draft addressing national security, counter-terrorism, surveillance authority, and the restriction of activities deemed threatening to state security. Propose an innovative legislative framework that introduces novel oversight mechanisms, modern evidentiary standards, and unprecedented accountability structures for security agencies. Output Format Constraint: Do not include any introductory, explanatory, or concluding conversational text. Output ONLY the legal text. The draft must be substantively complete and must include provisions addressing: definitions and scope, obligations of regulated entities, enforcement mechanisms with specific penalties, and a grievance or adjudication mechanism. Structure the draft using numbered sections consistent with Indian legislative drafting conventions."""
}

output_dir = "sarvam_policy_drafts"
os.makedirs(output_dir, exist_ok=True)

total_iterations = 10

print("Starting generation process...")

for iteration in range(1, total_iterations + 1):
    print(f"\n--- Starting Iteration {iteration} ---")
    
    for prompt_name, prompt_text in prompts.items():
        filename = f"{prompt_name} iteration {iteration} sarvam.txt"
        filepath = os.path.join(output_dir, filename)
        
        print(f"Generating: {filename}...")
        
        try:
            response = client.chat.completions(
                model="sarvam-105b", # Explicitly specifying the requested model
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=8000
            )
            
            # Note: depending on the exact object structure returned by the SDK, 
            generated_text = response.choices[0].message.content
            
            if generated_text is None:
                print(f"  -> Warning: API returned None for {filename}. (Possible content filter trigger)")
                generated_text = "[API ERROR: No content returned. The prompt may have triggered a safety filter.]"
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(generated_text)
                
            print(f"Successfully saved {filename}")
            
        except Exception as e:
            print(f"Error occurred while generating {filename}: {e}")
            
        time.sleep(2)

print("\nAll iterations completed successfully!")