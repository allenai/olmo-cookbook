import os
from pathlib import Path
import jinja2

CATEGORIES = [
    "adult",
    "art_design",
    "crime_law",
    "education_jobs",
    "entertainment",
    "fashion_beauty",
    "finance_business",
    "food_dining",
    "games",
    "hardware",
    "health",
    "history",
    "home_hobbies",
    "industrial",
    "literature",
    "politics",
    "religion",
    "science_tech",
    "social_life",
    "software",
    "software_dev",
    "sports_fitness",
    "transportation",
    "travel"
]

def main():
    # Get the directory where this script is located
    base_dir = Path(__file__).parent
    scripts_dir = base_dir / "scripts"
    
    # Create scripts directory if it doesn't exist
    scripts_dir.mkdir(exist_ok=True)
    
    # Read the template file
    template_path = base_dir / "tokenize.sh"
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Create Jinja2 environment and template
    env = jinja2.Environment()
    template = env.from_string(template_content)
    
    # Generate script for each category
    for category in CATEGORIES:
        # Render the template with the category
        rendered_content = template.render(category=category)
        
        # Write to the output file
        output_path = scripts_dir / f"tokenize_{category}.sh"
        with open(output_path, 'w') as f:
            f.write(rendered_content)
        
        # Make the script executable
        os.chmod(output_path, 0o755)
        
        print(f"Generated script: {output_path}")
    
    print(f"\nGenerated {len(CATEGORIES)} tokenization scripts in {scripts_dir}")

if __name__ == "__main__":
    main()