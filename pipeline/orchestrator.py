from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict

from pipeline.config import PipelineConfig
from pipeline.extractors.base import ExtractedContent
from pipeline.generators import get_generator
from pipeline.generators.base import BaseGenerator, GeneratedInstructions
from pipeline.memory.pinecone_service import PineconeMemory
from pipeline.utils.logging import get_logger
from pipeline.utils.models import get_model_for_role

if TYPE_CHECKING:
    from logging import Logger


class AgentRole(str, Enum):
    """Roles for specialized agents in the orchestration pipeline."""

    PLANNER = "planner"
    ENGINEER = "engineer"
    TESTER = "tester"
    REVIEWER = "reviewer"
    THINKER = "thinker"


class WorkflowStep(TypedDict, total=False):
    """Definition of a single step in a workflow."""

    name: str
    role: str
    action: str
    input_from: str | None
    prompt_template: str
    max_retries: int


class WorkflowConfig(TypedDict, total=False):
    """Configuration for a multi-agent workflow."""

    name: str
    description: str
    steps: list[WorkflowStep]
    parallel_steps: list[list[str]]


@dataclass
class AgentContext:
    """Context passed between agents during workflow execution."""

    original_content: ExtractedContent
    intermediate_results: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)

    def add_result(self, step_name: str, result: str) -> None:
        """Add a result from a workflow step."""
        self.intermediate_results[step_name] = result

    def get_result(self, step_name: str) -> str | None:
        """Get a result from a previous step."""
        return self.intermediate_results.get(step_name)


@dataclass
class WorkflowResult:
    """Result of a complete workflow execution."""

    workflow_name: str
    success: bool
    steps_completed: list[str]
    final_output: str
    execution_time_seconds: float
    agent_outputs: dict[str, str]
    errors: list[str] = field(default_factory=list)




class AgentOrchestrator:
    """Orchestrates multiple AI agents for complex workflows."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger: Logger = get_logger(__name__)
        self._generators: dict[AgentRole, BaseGenerator] = {}
        self.memory = PineconeMemory(config)

    def _get_generator_for_role(self, role: AgentRole) -> BaseGenerator:
        """Get or create a generator configured for a specific role."""
        if role not in self._generators:
            # Get the best model for this role
            model_name = get_model_for_role(role.value)
            self.logger.info(f"Using model {model_name} for role {role.value}")

            # Create a config copy with the role-specific model
            role_config = self.config.model_copy(deep=True)
            role_config.generator.model = model_name

            self._generators[role] = get_generator(role_config)

        return self._generators[role]

    async def execute_workflow(
        self, workflow: WorkflowConfig, content: ExtractedContent
    ) -> WorkflowResult:
        """Execute a complete multi-agent workflow."""
        self.logger.info(f"Starting workflow: {workflow.get('name', 'unnamed')}")

        context = AgentContext(original_content=content)
        steps_completed = []
        agent_outputs = {}
        errors = []

        steps = workflow.get("steps", [])

        for step in steps:
            step_name = step.get("name", "unnamed_step")
            role_str = step.get("role", "engineer")
            role = AgentRole(role_str.lower())

            self.logger.info(f"Executing step: {step_name} with role: {role.value}")

            try:
                result = await self._execute_step(step, role, context)
                context.add_result(step_name, result)
                agent_outputs[step_name] = result
                steps_completed.append(step_name)

            except Exception as e:
                error_msg = f"Step {step_name} failed: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)

                if step.get("max_retries", 0) > 0:
                    # Could implement retry logic here
                    pass

        # Calculate execution time
        execution_time = (datetime.now() - context.start_time).total_seconds()

        # Get final output (last step's result or combined)
        final_output = ""
        if agent_outputs:
            final_output = list(agent_outputs.values())[-1]
            
            # Store in memory
            if self.memory.enabled:
                self.memory.upsert(
                    content=final_output,
                    source_file=content.file_name,
                    metadata={"workflow": workflow.get("name"), "type": "workflow_result"}
                )

        return WorkflowResult(
            workflow_name=workflow.get("name", "unnamed"),
            success=len(errors) == 0,
            steps_completed=steps_completed,
            final_output=final_output,
            execution_time_seconds=execution_time,
            agent_outputs=agent_outputs,
            errors=errors,
        )

    async def _execute_step(
        self, step: WorkflowStep, role: AgentRole, context: AgentContext
    ) -> str:
        """Execute a single workflow step."""
        generator = self._get_generator_for_role(role)

        # Build the step prompt
        prompt_template = step.get("prompt_template", "")
        input_from = step.get("input_from")

        # Get input from previous step if specified
        previous_output = ""
        if input_from:
            previous_output = context.get_result(input_from) or ""

        # Create a modified content with the step-specific prompt
        step_content = ExtractedContent(
            content=context.original_content.content,
            summary=context.original_content.summary,
            file_path=context.original_content.file_path,
            file_type=context.original_content.file_type,
            file_size_bytes=context.original_content.file_size_bytes,
            modified_time=context.original_content.modified_time,
            structure=context.original_content.structure,
            metadata={
                **context.original_content.metadata,
                "step_prompt": prompt_template,
                "previous_output": previous_output,
            },
        )

        result: GeneratedInstructions = await generator.generate(step_content)
        return result.instructions

    async def execute_parallel_steps(
        self, steps: list[WorkflowStep], context: AgentContext
    ) -> dict[str, str]:
        """Execute multiple steps in parallel."""

        async def run_step(step: WorkflowStep) -> tuple[str, str]:
            role_str = step.get("role", "engineer")
            role = AgentRole(role_str.lower())
            result = await self._execute_step(step, role, context)
            return (step.get("name", "unnamed"), result)

        tasks = [run_step(step) for step in steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = {}
        for result in results:
            if isinstance(result, tuple):
                outputs[result[0]] = result[1]
            else:
                self.logger.error(f"Parallel step failed: {result}")

        return outputs

    def extract_json_from_output(self, output: str) -> dict[str, Any] | None:
        """Extract JSON block from agent output."""
        try:
            # Look for JSON between code blocks
            match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # Fallback to finding anything that looks like a JSON object
            # Finding the first { and the last } in the string
            start = output.find('{')
            end = output.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = output[start:end+1]
                return json.loads(json_str)
        except ValueError:
            pass
        return None

    def create_code_review_workflow(self) -> WorkflowConfig:
        """Create a pre-defined code review workflow."""
        return {
            "name": "code_review",
            "description": "Multi-agent code review workflow",
            "steps": [
                {
                    "name": "analyze",
                    "role": "planner",
                    "action": "analyze",
                    "prompt_template": "Analyze this code and identify key areas for review.",
                    "max_retries": 1,
                },
                {
                    "name": "review",
                    "role": "engineer",
                    "action": "review",
                    "input_from": "analyze",
                    "prompt_template": "Review the code focusing on: {previous_output}",
                    "max_retries": 1,
                },
                {
                    "name": "test_plan",
                    "role": "tester",
                    "action": "plan_tests",
                    "input_from": "review",
                    "prompt_template": "Create test cases based on the review: {previous_output}",
                    "max_retries": 1,
                },
            ],
            "parallel_steps": [],
        }

    def create_data_analysis_workflow(self) -> WorkflowConfig:
        """Create a pre-defined data analysis workflow."""
        return {
            "name": "data_analysis",
            "description": "Multi-agent data analysis workflow",
            "steps": [
                {
                    "name": "explore",
                    "role": "thinker",
                    "action": "explore",
                    "prompt_template": "Explore this data and identify patterns, anomalies, and insights.",
                    "max_retries": 1,
                },
                {
                    "name": "analyze",
                    "role": "engineer",
                    "action": "analyze",
                    "input_from": "explore",
                    "prompt_template": "Perform detailed analysis on: {previous_output}",
                    "max_retries": 1,
                },
                {
                    "name": "summarize",
                    "role": "planner",
                    "action": "summarize",
                    "input_from": "analyze",
                    "prompt_template": "Create an executive summary: {previous_output}",
                    "max_retries": 1,
                },
            ],
            "parallel_steps": [],
        }
    def create_startup_application_workflow(self) -> WorkflowConfig:
        """Create a pre-defined startup application workflow."""
        return {
            "name": "startup_application",
            "description": "Multi-agent startup application extraction workflow",
            "steps": [
                {
                    "name": "analyze_startup",
                    "role": "planner",
                    "action": "analyze",
                    "prompt_template": "Analyze this startup document and identify core themes for a hacker house application.",
                    "max_retries": 1,
                },
                {
                    "name": "extract_form_data",
                    "role": "engineer",
                    "action": "extract",
                    "input_from": "analyze_startup",
                    "prompt_template": "Extract concise form-ready fields (founder, mission, tech stack) from: {previous_output}",
                    "max_retries": 2,
                },
                {
                    "name": "refine_pitch",
                    "role": "reviewer",
                    "action": "optimize",
                    "input_from": "extract_form_data",
                    "prompt_template": "Refine the extraction for a hacker house application. Output final pitch and a JSON block for auto-filling.",
                    "max_retries": 1,
                },
            ],
            "parallel_steps": [],
        }
