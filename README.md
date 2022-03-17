# SCD
Master Thesis project

# Install Requirements
PYTHON-PATH -m pip install -r requirements.txt

# Accessing Aim-Logger when Training on Remote
ssh -L AAAA:ip:BBBB  user@ip
When opening http://localhost:AAAA/ on local computer it will connect to the remote port BBBB. AAAA and BBBB can be the same.

ssh -L 43800:<ip>:43800 user@<ip>
