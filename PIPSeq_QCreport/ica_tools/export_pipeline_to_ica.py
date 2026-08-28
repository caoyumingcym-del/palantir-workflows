#!/usr/bin/env python3
"""Import/re-import PIPSeq_QCreport into ICA at the current git commit.

Adapted from SingleCell/PIPseqDownsample/ica_tools/export_pipeline_to_ica.py
in this same repo -- same API calls, same flow, simplified for one entrypoint
instead of two. Exists because ICA's git-based pipeline import pins to a
specific commit (not a branch) and the commit isn't editable from the ICA
UI's pipeline-edit screen -- re-running this script is the update mechanism,
via the same importGitPipeline API call the UI's "Import from Git" wizard
uses under the hood.

Every run creates a NEW pipeline entry in ICA (name includes the short commit
hash, so it's always unique) rather than updating one in place -- that's
also true of the sibling script this is based on.

Before running, fill in:
  - ica_tools/ica_common.py:PROJECT_NAMES_AND_IDS -- your ICA project(s).
  - GIT_CREDENTIAL_UUID below -- an ICA git credential authorized to read
    REPOSITORY_URL. Project Settings -> Credentials (or similar) in the ICA
    UI; create one if none exists yet for this repository. If
    REPOSITORY_URL is a public repo, this may not be required at all --
    worth trying with an empty credential first if you hit permission
    trouble creating one.
  - ~/.icav2/api_key.txt -- your ICA API key (see ica_common.load_api_key).

Usage:
    python3 export_pipeline_to_ica.py
"""

import datetime
import os
import subprocess
import time

import requests

from ica_common import API_URL, PROJECT_NAMES_AND_IDS, load_api_key, prompt_choice

# ---------------------------------------------------------------- fill in
REPOSITORY_URL = 'https://github.com/caoyumingcym-del/palantir-workflows'
GIT_CREDENTIAL_UUID = ''  # <-- an ICA git credential ID; see module docstring
# ---------------------------------------------------------------------------

MAIN_FILE_PATH = 'PIPSeq_QCreport/main.nf'
NEXTFLOW_CONFIG_PATH = 'PIPSeq_QCreport/nextflow.config'
INPUT_FORM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'inputForm.json'
)

api_url = API_URL
ica_api_key = load_api_key()

current_git_commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
current_git_commit_id_short = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
current_git_commit_message = subprocess.check_output(['git', 'log', '-1', '--pretty=%B']).decode().strip().split('\n')[0].strip()

if not GIT_CREDENTIAL_UUID:
    print('ERROR: GIT_CREDENTIAL_UUID is not set at the top of this script.')
    print('Find or create one in the ICA UI under this project\'s git')
    print(f'credentials, authorized to read {REPOSITORY_URL}, then paste its')
    print('ID in. If the repository is public, try an empty string there')
    print('first -- it may not be required at all.')
    exit(1)

if not os.path.isfile(INPUT_FORM_PATH):
    print(f'ERROR: input form file not found at {INPUT_FORM_PATH}')
    exit(1)

if not PROJECT_NAMES_AND_IDS:
    print('ERROR: ica_common.py:PROJECT_NAMES_AND_IDS is empty. Add your ICA')
    print('project name -> project ID there first.')
    exit(1)

pipeline_name = f'PIPSeq_QCreport_{current_git_commit_id_short}'

print('Exporting pipeline with the following data:')
print(f'  Pipeline name: {pipeline_name}')
print(f'  Pipeline version: {current_git_commit_id_short}')
print(f'  Commit: {current_git_commit_id}')
print(f'  Repository URL: {REPOSITORY_URL}')
print(f'  Main file path: {MAIN_FILE_PATH}')
print(f'  Nextflow config path: {NEXTFLOW_CONFIG_PATH}')
print(f'  Git credential UUID: {GIT_CREDENTIAL_UUID or "(none)"}')
print('')
project_names = list(PROJECT_NAMES_AND_IDS.keys())
project_choice = prompt_choice('Which project do you want to export to?', project_names)
if project_choice is None:
    exit(0)
project_id = PROJECT_NAMES_AND_IDS[project_names[project_choice]]

headers = {
    'X-API-Key': ica_api_key,
    'Accept': 'application/vnd.illumina.v4+json',
}

# For multipart/form-data, use the files parameter with (None, value) tuples.
# This forces requests to send as multipart/form-data instead of
# application/x-www-form-urlencoded.
files = {
    'language': (None, 'NEXTFLOW'),
    'code': (None, pipeline_name),
    'description': (None, f'Pipeline exported on {datetime.date.today().isoformat()}: {current_git_commit_message}'),
    'defaultStorageType': (None, 'Small'),
    'proprietary': (None, 'false'),
    'version': (None, current_git_commit_id_short),
    'gitCredentialId': (None, GIT_CREDENTIAL_UUID),
    'commitId': (None, current_git_commit_id),
    'repositoryUrl': (None, REPOSITORY_URL),
    'mainFilePath': (None, MAIN_FILE_PATH),
    'configFilePath': (None, NEXTFLOW_CONFIG_PATH),
}

response = requests.post(f'{api_url}/projects/{project_id}/pipelines:importGitPipeline', headers=headers, files=files)
if not response.ok:
    print(f'ERROR: import failed (HTTP {response.status_code}): {response.json().get("detail", response.text)}')
    exit(1)
pipeline_id = response.json()['id']
print(f'Import scheduled successfully (HTTP {response.status_code}). Pipeline ID: {pipeline_id}')

# The git pipeline import runs asynchronously (status starts as 'Importing').
# The input form can only be uploaded once ICA has finished parsing the repo
# and the pipeline reaches 'Draft'.
poll_interval_s = 5
max_attempts = 120  # 10 minutes
FAILURE_STATUSES = {'Import Failed', 'Import Incomplete', 'Import Cancelling'}

print('')
print(f'Waiting for pipeline {pipeline_id} to reach Draft status...')
status = None
for attempt in range(1, max_attempts + 1):
    poll_response = requests.get(f'{api_url}/projects/{project_id}/pipelines/{pipeline_id}', headers=headers, timeout=30)
    poll_response.raise_for_status()
    status = poll_response.json()['pipeline']['statusAsString']
    print(f'  [{attempt}/{max_attempts}] status: {status}')
    if status == 'Draft':
        break
    if status in FAILURE_STATUSES:
        print(f'ERROR: pipeline import ended in status "{status}"; not uploading input form.')
        exit(1)
    time.sleep(poll_interval_s)
else:
    print(f'ERROR: pipeline did not reach Draft status within {max_attempts * poll_interval_s}s (last status: {status}); not uploading input form.')
    exit(1)

print('')
print(f'Uploading input form from {INPUT_FORM_PATH}...')
# The inputForm/inputFormFile endpoint only recognizes the v3 API version --
# sending the v4 Accept header used for the other endpoints above gets
# rejected with "Invalid Accept Header".
input_form_headers = {**headers, 'Accept': 'application/vnd.illumina.v3+json'}
with open(INPUT_FORM_PATH, 'rb') as input_form_file:
    form_response = requests.put(
        f'{api_url}/projects/{project_id}/pipelines/{pipeline_id}/inputForm/inputFormFile',
        headers=input_form_headers,
        files={'content': ('inputForm.json', input_form_file, 'application/json')},
    )
if not form_response.ok:
    print(f'ERROR: input form upload failed (HTTP {form_response.status_code}): {form_response.json().get("detail", form_response.text)}')
    exit(1)
print(f'Input form uploaded successfully (HTTP {form_response.status_code}).')
print('Done.')
