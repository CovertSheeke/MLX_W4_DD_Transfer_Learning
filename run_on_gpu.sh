# ssh -p [PORT] -i ~/.ssh/mli_computa root@[IP]
# Then copy/paste this script into the remote machine

apt-get update
apt-get install git
#apt-get install git-lfs
apt-get install tmux -y
apt-get install nvtop -y
git lfs install
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

cd ~
git clone https://github.com/CovertSheeke/MLX_W4_DD_Transfer_Learning
cd MLX_W4_DD_Transfer_Learning
uv sync

# Change to suit your needs! 
git config --global user.email "exampleuser@my_mail.com"
git config --global user.name "example-user"

# You can generate a new token at https://github.com/settings/personal-access-tokens
# => Select only this repository
# => Select Read and Write access to Contents (AKA Code)

# Launches a new tmux session (with name sweep) the name is optional!
# This session can survive even if you disconnect from SSH
# => Ctrl+B enters command mode in tmux (then release ctrl)
# ==> Ctrl+B (unclick Ctrl) then D detaches from the current tmux session
# => Discover existing sessions with tmux ls
# => Reattach to the last session with tmux a (short for attach)
# => Reattach with tmux attach -t 0
# => Scroll with Ctrl+B [ then use the arrow keys or mouse to scroll up and down. Leave scroll mode with Esc or q

tmux new -s mlx

# Check GPU usage with the nvtop command

# Now run a script, e.g.
# uv run -m model.continue_train
# uv run -m model.models