"""Shared helpers for the ICA API scripts in this directory (export_pipeline_to_ica.py).

Mirrors SingleCell/PIPseqDownsample/ica_tools/ica_common.py in this same repo --
same API base URL and conventions, so anyone already using that one will
recognize this immediately. Only the project mapping differs, since that's
specific to whichever ICA project *you* export this pipeline into.
"""

import os

API_URL = 'https://ica.illumina.com/ica/rest/api'

# Fill in with your own ICA project name(s) -> project ID. Find a project ID
# either in its URL in the ICA web UI, or via `icav2 projects list` if you
# have the icav2 CLI configured.
PROJECT_NAMES_AND_IDS = {
    # 'My ICA Project': '00000000-0000-0000-0000-000000000000',
}


def load_api_key():
    api_key_path = os.path.expanduser('~/.icav2/api_key.txt')
    if not os.path.isfile(api_key_path):
        raise SystemExit(
            f'ERROR: no API key found at {api_key_path}. Generate one in the '
            f'ICA web UI (usually under your user/profile settings -> API '
            f'keys) and save just the key string to that file.'
        )
    return open(api_key_path).read().strip()


def prompt_choice(prompt, options):
    """Prints a numbered list of options and asks the user to pick one.

    Returns the chosen 0-based index, or None if the user aborted.
    """
    print(prompt)
    for i, option in enumerate(options, start=1):
        print(f'  {i}. {option}')
    choice = input('Enter the number of your choice (or anything else to abort): ')
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(options):
        return None
    return int(choice) - 1
