import sys
import argparse
import os
from pathlib import Path
from importlib import import_module
import importlib.util
import traceback

# Ensure project root is in the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.resolve()))

# Map workflows that share the same runner
shared_runner_map = {
    "ml_batch_generation": "ml_data_generation",
    "ml_batch_generation_restart": "ml_data_generation",
    "voxelize_into_json": "ml_data_generation",
    "voxelize_into_json_test": "ml_data_generation",
    "ml_generation_confirmation": "ml_data_generation",
    "visualize_voxelized_domain": "ml_data_generation",
    # Add more mappings here if needed
}

def run_workflow(config, workflow_name):
    # Determine the runner module name - default to workflow_name if not
    # in shared_runner_map
    runner_name = shared_runner_map.get(workflow_name, workflow_name)

    try:
        module_name = f"workflow_runner.{runner_name}"
        print(f"🧪 Trying import: {module_name}")

        spec = importlib.util.find_spec(module_name)
        print("🧪 Import spec:", spec)
        runner_module = import_module(f"workflow_runner.{runner_name}")
        runner_module.run(config)
    except ModuleNotFoundError as e:
        print(f"❌ Runner module 'workflow_runner.{runner_name}' not found:\n{e}")
        raise
    except AttributeError as e:
        print(f"❌ 'run(config)' function not found in workflow_runner.{runner_name}:\n{e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error while running workflow '{workflow_name}':\n{e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run a workflow.")
    parser.add_argument(
        "--workflow", type=str, required=True,
        help="Workflow module name in 'workflows/' (e.g., standardize_json)"
    )
    parser.add_argument("--set", nargs="*", help="Override config values. Format: key=value")

    args = parser.parse_args()
    workflow_name = args.workflow

    try:
        print("🧪 Current working dir:", os.getcwd())
        print("🧪 Contents of workflow_runner:", list(Path("workflow_runner").glob("*.py")))
        workflow_module = import_module(f"workflows.{workflow_name}")
        config = workflow_module.get_config()
        if args.set:
            for item in args.set:
                if "=" not in item:
                    raise ValueError(f"Invalid override: {item} (must be key=value)")
                key, value = item.split("=", 1)
                if key in config and isinstance(config[key], bool):
                    config[key] = value.lower() in ("true", "1", "yes")
                elif key in config and isinstance(config[key], (int, float)):
                    config[key] = type(config[key])(value)
                else:
                    config[key] = value  # Leave as string if not known
        print(f"🚀 Running workflow: {workflow_name}")
        run_workflow(config, workflow_name)

    except ModuleNotFoundError:
        print(f"❌ Workflow module 'workflows.{workflow_name}' not found.")
        traceback.print_exc()
    except AttributeError:
        print(f"❌ 'get_config()' not found in workflows.{workflow_name}.")
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Unexpected error while loading workflow '{workflow_name}':\n{e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()