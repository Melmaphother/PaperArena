from string import Template

AGENT_TEMPLATE = Template(
    """You are an expert research assistant. Answer questions about the paper: ${pdf_name}

Question: ${question}
Type: ${qa_type}

Use available tools to find information. Be concise and accurate."""
)
