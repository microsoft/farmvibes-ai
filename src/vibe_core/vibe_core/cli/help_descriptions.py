ADD_ONNX_HELP = """\
Adds an Onnx model <model-file> to the TerraVibes cluster. After being added, one can use this \
model in the ops compute_onnx and compute_onnx_from_sequence (by setting the parameter model_file \
accordingly)."""

ADD_SECRET_HELP = """\
Adds a secret with a key <key> and value <value>.

Secrets are used in ops with the @SECRET parameter. For example,\
"@SECRET(eywa-secrets, pc-sub-key)" in which "pc-sub-key" is the key and\
eywa-secrets is the key-vault. Key-vaults are only required for an Azure\
instalation. For this local farmvibes.ai instalation, the key-vault can be any\
non-empty string."""

DELETE_SECRET_HELP = "Deletes secret with the key <key>."

DESTROY_HELP = """\
Stops the FarmVibes.AI cluster and deletes all traces of it from the user's docker\
installation."""

RESTART_HELP = "Restarts the FarmVibes.AI cluster."

SETUP_HELP = """\
Sets up a new local kubernetes cluster with FarmVibes.AI running in it.\

This will create a cluster with your local kubernetes (e.g., `k3d`), download container images \
from the FarmVibes.AI container registry, and load them into the cluster."""

START_HELP = """\
Starts an existing FarmVibes.AI cluster. This also starts the kubernetes control\
plane, and helper services, such as a redis instance, that support the execution\
of FarmVibes.AI."""

STATUS_HELP = "Shows the FarmVibes.AI cluster's status (if one exists)."

STOP_HELP = """\
Stops an existing FarmVibes.AI cluster. This also stops the kubernetes control\
plane, and helper services, such as a redis instance, that support the execution\
of FarmVibes.AI."""

UPDATE_HELP = """\
Upgrades the vibe_core library to the version referenced in the repository, and\
pulls new images from the remote container registry and adds the downloaded\
images to an existing FarmVibes.AI cluster.\

After images are successfully pulled, restarts the FarmVibes.AI services so\
that they use these new images."""
