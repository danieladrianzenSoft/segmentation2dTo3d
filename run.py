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
    # Automatically collect all workflow scripts (excluding __init__.py and non-py files)
    workflow_dir = Path(__file__).resolve().parent / "workflows"
    available_workflows = sorted([
        f.stem for f in workflow_dir.glob("*.py")
        if f.name != "__init__.py" and not f.name.startswith("_")
    ])

    # Initial parser just to handle --list early
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--list", action="store_true", help="List all available workflows and exit")
    args, remaining_argv = pre_parser.parse_known_args()

    # Handle --list before any required args
    if args.list:
        print("Available workflows:")
        for wf in available_workflows:
            print("  -", wf)
        sys.exit(0)

    # Main parser
    parser = argparse.ArgumentParser(
        description="Run a workflow module from the 'workflows/' folder.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--set", nargs="*", metavar="key=value",
        help="Override config values dynamically (e.g. --set input_path=foo.png threshold=2)"
    )

    parser.add_argument(
        "--workflow", type=str, required=True,
        metavar="WORKFLOW",
        help=(
            "workflow module name to run from 'workflows/'.\n"
            f"  - " + "\n  - ".join(available_workflows)
        )
    )

    # Re-parse
    args = parser.parse_args(remaining_argv)
    workflow_name = args.workflow
    if workflow_name not in available_workflows:
        print(f"Invalid workflow: '{args.workflow}'\n")
        print("Available workflows:")
        for wf in available_workflows:
            print(f"  - {wf}")
        sys.exit(1)

    try:
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