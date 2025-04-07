#!/usr/bin/env python
import sys
import os
import shutil
import argparse
def main():
    parser = argparse.ArgumentParser(description='Create a configuration file from template.')
    parser.add_argument('pool', help='The pool value to use in the configuration')
    parser.add_argument('mix', help='The mix value to use in the configuration')
    args = parser.parse_args()
    
    pool = args.pool
    mix = args.mix
    
    template_path = "configs/olmo-cookbook-1b-5xC-template.yaml"
    output_path = f"configs/olmo-cookbook-1b-5xC-{pool}-{mix}.yaml"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read the template
    with open(template_path, 'r') as template_file:
        template_content = template_file.read()
    
    # Replace placeholders
    output_content = template_content.format(POOL=pool, MIX=mix)
    
    # Write the new config file
    with open(output_path, 'w') as output_file:
        output_file.write(output_content)
    
    print(f"Created config file: {output_path}")

if __name__ == "__main__":
    main()
