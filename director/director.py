from langchain_core.language_models.chat_models import BaseChatModel
from langchain_experimental.generative_agents import (
    GenerativeAgent
)
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import operator
# Creates a director agent that creates randomized world events and 
# stores them in the memory of the character agent.


tools = []

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a director agent that creates randomized world events for the characters.
        You will be given a list of characters, traits and statuses.
        You will also be given a list of tools to help you randomize the events.
        You will create a world event that is relevant to the characters, traits and statuses.
        Available tools: {tools_names}
    """),
    ("human", "Give me a randomized event for this character: {input}"),
    ("ai", "{agent_scratchpad}")
])

class DirectorState(TypedDict):
    events: Annotated[List[BaseMessage], operator.add]
    input: str
    structured_response: str
    remaining_steps: int
    is_last_step: bool
    tool_outputs: List[str]
    tool_called: List[str]

def get_director(llm: BaseChatModel, list_of_characters: List[GenerativeAgent], tools, prompt):
    def call_model(state: DirectorState):
        # Prepare the response
        events = state['events']
        input_text = state['input']

        # Bind tools
        llm_with_tools = llm.bind_tools(tools)

        tool_names = [tool.name for tool in tools]

        formatted_prompt = prompt.format_messages(
            tools_names=", ".join(tool_names),
            agent_scratchpad=state.get('structured_response', ''),
            input=events
        )

        # Invoke the model
        response = llm_with_tools.invoke(formatted_prompt)
        
        # Handle tool calls if present
        if response.tool_calls:
            tool_outputs = []
            for tool_call in response.tool_calls:
                tool = next((t for t in tools if t.name == tool_call['name']), None)
                if tool:
                    try:
                        tool_output = tool.invoke(tool_call['args'])
                        tool_outputs.append(
                            ToolMessage(
                                content=str(tool_output), 
                                tool_call_id=tool_call['id']
                            )
                        )

                        # format tool output
                        print("Tool Used:", tool.name)
                        # if tool.name == 'WorkorderStatus':
                        #     woDict = tool_output['Retrieved Workorder Data:']
                            
                        #     state["tool_called"].append("WorkOrderStatus")
                        #     state["tool_outputs"].append({
                        #         "Status" : woDict['workorder status'],
                        #         "Technicians" : woDict['workorder technicians'],
                        #         "Equipment" : woDict['workorder equipment'],
                        #         "Parts / Purchases" : woDict['workorder parts']
                        #     })
                    except Exception as e:
                        tool_outputs.append(
                            ToolMessage(
                                content=f"Error using tool {tool_call['name']}: {str(e)}",
                                tool_call_id=tool_call['id']
                            )
                        )
            
            # Update events with tool outputs
            events.extend(tool_outputs)

        # Determine remaining steps and last step status
        remaining_steps = state.get('remaining_steps', 3) - 1
        is_last_step = remaining_steps <= 0
        
        return {
            "events": [response],
            "input": input_text,
            "remaining_steps": remaining_steps,
            "is_last_step": is_last_step,
            "tool_outputs": state["tool_outputs"],
            "tool_called": state["tool_called"]
        }

    # Create the workflow
    workflow = StateGraph(DirectorState)
    workflow.add_node("agent", call_model)
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        lambda state: END if state['is_last_step'] else "agent",
        {
            "agent": "agent",
            END: END
        }
    )

    return workflow.compile()
