#!/bin/sh

DOCKER_VERSION=24.0.2
VSCODE_HOME=/home/vscode

if [ -d /user-home/.ssh ]; then
    echo "Making user ssh available in container..."
    mkdir -p $VSCODE_HOME/.ssh
    chmod 0700 $VSCODE_HOME/.ssh
    for f in /user-home/.ssh/*
    do
        cp "$f" $VSCODE_HOME/.ssh/"$(basename "$f")"
        chmod 0600 $VSCODE_HOME/.ssh/"$(basename "$f")"
    done
fi

# If the user has a git config file, copy it
if [ -f /user-home/.gitconfig ]; then
    echo "Copying user .gitconfig..."
    cp /user-home/.gitconfig $VSCODE_HOME/.gitconfig
    echo "Enabling HTTP use path, in case the user cloned with HTTP"
    git config --global credential.useHttpPath true
fi

if [ "$(stat -c '%u' .)" != "$UID" ]; then
    echo "The permissions of the current directory differ from the current user,"
    echo "which means we're probably running in Docker under a Windows host..."
    echo "Adding the current directory to the git safe directory list"
    git config --global --add safe.directory /workspaces/TerraVibes
fi

sudo mkdir /opt/venv
sudo chown vscode /opt/venv
/opt/conda/bin/python3 -m venv --system-site-packages /opt/venv || exit 1
/opt/venv/bin/pip install --upgrade pip

if [[ "$(uname -a)" == *"WSL2"* ]]; then
    # We're either in WSL2 or in a Windows host
    echo "If we're on a Windows host, we need to convert files to unix mode..."
    find cli scripts -type f -exec dos2unix --allow-chown {} \;
fi

sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
zsh -c "zstyle ':omz:update' mode auto"
zsh -c "zstyle ':omz:update' verbose minimal"
git clone https://github.com/zsh-users/zsh-autosuggestions.git ~/.oh-my-zsh/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $ZSH_CUSTOM/plugins/zsh-syntax-highlighting
git clone --depth 1 -- https://github.com/marlonrichert/zsh-autocomplete.git $ZSH_CUSTOM/plugins/zsh-autocomplete
sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting zsh-autocomplete)/g' ~/.zshrc
echo "export LD_LIBRARY_PATH=/opt/conda/lib:\$LD_LIBRARY_PATH" >> ~/.zshrc
echo "export LD_LIBRARY_PATH=/opt/conda/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

/opt/venv/bin/pip install --upgrade pyright
/opt/venv/bin/pip install --upgrade "pytest" "anyio[trio]"
sed -e '1,/dependencies:/d' < resources/envs/dev.yaml | \
    sed 's/-//' | \
    xargs /opt/venv/bin/pip install
eval $(grep 'terravibes_packages=' < "scripts/setup_python_develop_env.sh")
for package in $terravibes_packages
do
    /opt/venv/bin/pip install -e src/$package
done

sudo mkdir -p /opt/terravibes/ops
sudo ln -sf $(pwd)/op_resources /opt/terravibes/ops/resources
sudo mkdir /app
sudo ln -sf $(pwd)/ops /app/ops
sudo ln -sf $(pwd)/workflows /app/workflows