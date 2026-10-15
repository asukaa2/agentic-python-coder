import argparse
import os
import sys
from pathlib import Path
import gradio as gr

from agentic_python_coder.runner import get_system_prompt_path, save_conversation_log
from agentic_python_coder.agent import (
    create_coding_agent,
    run_agent,
    get_final_response,
)
from agentic_python_coder.project_md import (
    parse_project_file,
    check_packages_available,
    create_project_prompt,
)
from agentic_python_coder.llm import DEFAULT_MODEL
from agentic_python_coder import __version__


def parse_args():
    """Parse command-line arguments for launching the Gradio server."""
    parser = argparse.ArgumentParser(description="Gradio web interface for the Python coding agent.")
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=7860,
        help="Port to run the Gradio server on (default: 7860)"
    )
    parser.add_argument(
        "--host",
        "-H",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a public shareable link for the Gradio app"
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        help="Automatically open the app in your default web browser"
    )
    return parser.parse_args()


def create_app():
    with gr.Blocks() as demo:
        gr.Markdown(f"# Coder - AI-powered Python Coding Assistant (v{__version__})")
        gr.Markdown("Configure your agent settings on the left and interact with it through the chat interface.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")
                model_input = gr.Textbox(
                    label="Model",
                    value=DEFAULT_MODEL,
                    info="Model name or path to custom JSON config"
                )
                working_dir_input = gr.Textbox(
                    label="Working Directory",
                    value=str(Path.cwd()),
                    info="Directory where the agent executes code"
                )
                api_key_input = gr.Textbox(
                    label="OpenRouter API Key",
                    type="password",
                    info="Overrides default environment configuration"
                )
                project_input = gr.Textbox(
                    label="Project Markdown Path",
                    placeholder="path/to/project.md",
                    info="Optional project spec with examples & packages"
                )
                packages_input = gr.Textbox(
                    label="Additional Packages (--with)",
                    placeholder="pandas, numpy",
                    info="Comma-separated list of extra packages"
                )
                todo_checkbox = gr.Checkbox(
                    label="Enable Todo Tracking (--todo)",
                    value=False,
                    info="Enable todo_write tool for planning"
                )
                step_limit_input = gr.Number(
                    label="Step Limit",
                    value=200,
                    precision=0,
                    info="Maximum agent steps before stopping"
                )

                init_btn = gr.Button("Initialize / Reset Agent", variant="primary")
                status_box = gr.Textbox(label="Status", value="Agent not initialized.", interactive=False)

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Coding Agent Chat", height=500)
                msg = gr.Textbox(
                    label="Your Task / Message",
                    placeholder="Type your task or instructions here...",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

        # Session State variables
        agent_state = gr.State(None)
        all_messages_state = gr.State([])
        stats_state = gr.State({
            "tool_usage": {},
            "token_consumption": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "execution_time_seconds": 0,
        })
        working_dir_state = gr.State(Path.cwd())

        def initialize_agent(model, working_dir, api_key, project_path, packages_str, todo, step_limit):
            try:
                wd = Path(working_dir).resolve()
                wd.mkdir(parents=True, exist_ok=True)
                os.chdir(wd)

                with_packages = [p.strip() for p in packages_str.split(",")] if packages_str.strip() else None

                project_prompt = None
                if project_path.strip():
                    try:
                        packages, content = parse_project_file(project_path.strip())
                        unavailable = check_packages_available(packages) if packages else []
                        project_prompt = create_project_prompt(packages, content, unavailable)
                    except Exception as e:
                        return None, wd, [], {"tool_usage": {}, "token_consumption": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "execution_time_seconds": 0}, f"Error loading project: {e}"

                system_prompt = get_system_prompt_path(todo).read_text()

                agent = create_coding_agent(
                    working_directory=str(wd),
                    system_prompt=system_prompt,
                    model=model.strip() or DEFAULT_MODEL,
                    project_prompt=project_prompt,
                    with_packages=with_packages,
                    api_key=api_key.strip() if api_key.strip() else None,
                    todo=todo,
                    verbose=True,
                )

                initial_stats = {
                    "tool_usage": {},
                    "token_consumption": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                    "execution_time_seconds": 0,
                }

                return agent, wd, [], initial_stats, f"Agent successfully initialized in {wd}"
            except Exception as e:
                return None, Path.cwd(), [], {"tool_usage": {}, "token_consumption": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "execution_time_seconds": 0}, f"Error: {e}"

        init_btn.click(
            initialize_agent,
            inputs=[
                model_input,
                working_dir_input,
                api_key_input,
                project_input,
                packages_input,
                todo_checkbox,
                step_limit_input
            ],
            outputs=[agent_state, working_dir_state, all_messages_state, stats_state, status_box]
        )

        def respond(user_message, agent, working_dir, all_messages, stats):
            if not agent:
                history = chatbot.value + [[user_message, "Error: Agent is not initialized. Please click 'Initialize / Reset Agent' first."]]
                return history, agent, all_messages, stats

            if not user_message.strip():
                return chatbot.value, agent, all_messages, stats

            try:
                new_messages, run_stats = run_agent(agent, user_message, quiet=False)
                all_messages.extend(new_messages)

                # Update cumulative stats
                for tool, count in run_stats.get("tool_usage", {}).items():
                    stats["tool_usage"][tool] = stats["tool_usage"].get(tool, 0) + count

                for key in ["input_tokens", "output_tokens", "total_tokens"]:
                    stats["token_consumption"][key] += run_stats.get("token_consumption", {}).get(key, 0)

                stats["execution_time_seconds"] += run_stats.get("execution_time_seconds", 0)

                # Save conversation log
                save_conversation_log(working_dir, all_messages, stats)

                response_text = get_final_response(new_messages) or "Task completed (no textual response returned)."

                history = chatbot.value + [[user_message, response_text]]
                return history, agent, all_messages, stats
            except Exception as e:
                err_msg = f"Error during execution: {e}"
                history = chatbot.value + [[user_message, err_msg]]
                return history, agent, all_messages, stats

        submit_btn.click(
            respond,
            inputs=[msg, agent_state, working_dir_state, all_messages_state, stats_state],
            outputs=[chatbot, agent_state, all_messages_state, stats_state]
        ).then(lambda: "", None, msg)

        msg.submit(
            respond,
            inputs=[msg, agent_state, working_dir_state, all_messages_state, stats_state],
            outputs=[chatbot, agent_state, all_messages_state, stats_state]
        ).then(lambda: "", None, msg)

        clear_btn.click(lambda: [], None, chatbot)

    return demo


if __name__ == "__main__":
    args = parse_args()
    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser
    )
