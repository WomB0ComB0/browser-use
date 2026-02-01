from pipeline.utils.models import ModelOrchestrator, TaskRole


def test_discovery():
    print("Initializing ModelOrchestrator...")
    orchestrator = ModelOrchestrator()
    
    # Force clear cache for testing discovery
    orchestrator.clear_cache()
    
    print("\nDiscovering models...")
    for role in [TaskRole.PLANNER, TaskRole.ENGINEER, TaskRole.THINKER]:
        model = orchestrator.get_best_model_for_task(role)
        print(f"Role: {role.name:10} -> Model: {model}")

if __name__ == "__main__":
    test_discovery()
