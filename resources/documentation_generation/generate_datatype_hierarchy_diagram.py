import os
import subprocess
from typing import List

from jinja2 import Template

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
DOC_DIR = os.path.abspath(
    os.path.join(PROJECT_DIR, "docs", "source", "docfiles", "markdown", "data_types_diagram")
)
DATA_TYPES_PATH = os.path.abspath(
    os.path.join(PROJECT_DIR, "src", "vibe_core", "vibe_core", "data")
)
TEMPLATE_PATH = os.path.abspath(os.path.join(HERE, "templates", "datatype_hierarchy_template.md"))


def render_template(
    mermaid_diagram: str,
    output_path: str,
    template_path: str,
):
    """Load and render template given a data source"""

    with open(template_path) as f:
        t = Template(f.read())

    rendered_template = t.render(mermaid_diagram=mermaid_diagram)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w") as f:
        f.write(rendered_template)


def list_modules(module_path: str) -> List[str]:
    """List all modules in module_path"""

    paths = []
    for root, dirs, files in os.walk(module_path):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                paths.append(os.path.join(root, file))

    return paths


def build_data_type_diagrams(data_module_paths: List[str]):
    for path in data_module_paths:
        module_name = path.split("/")[-1].split(".")[0]
        subprocess.run(
            [
                "pyreverse",
                "-my",
                "-A",
                "-k",
                "-o",
                "mmd",
                "-p",
                f"{module_name}",
                path,
            ],
            check=True,
        )

        with open(f"classes_{module_name}.mmd") as f:
            mmd = f.read()
        render_template(mmd, os.path.join(DOC_DIR, f"{module_name}_hierarchy.md"), TEMPLATE_PATH)

        # Delete the generated mmd file with subprocess.run
        subprocess.run(["rm", f"classes_{module_name}.mmd"], check=True)


def main():
    data_module_paths = list_modules(DATA_TYPES_PATH)
    build_data_type_diagrams(data_module_paths)


if __name__ == "__main__":
    main()
