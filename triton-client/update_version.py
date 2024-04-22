import os
import toml

# Read the new version from the environment variable
new_version = os.environ.get('CI_PIPELINE_IID')

if new_version:
    # Read the existing pyproject.toml
    with open('pyproject.toml', 'r') as f:
        pyproject = toml.load(f)

    # Update the version in the pyproject dictionary
    pyproject['tool']['poetry']['version'] = new_version

    # Write the updated pyproject.toml back to the file
    with open('pyproject.toml', 'w') as f:
        toml.dump(pyproject, f)
else:
    print("No version provided in the environment variable NEW_VERSION.")
