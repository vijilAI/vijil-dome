# Evaluating and Protecting Agents powered by Groq

This tutorial accompanies our blog post on evaluating and protecting agents using Groq. The code creates a news agent that uses Groq's serverless inference to search the web for AI news, then demonstrates how to evaluate the agent with Vijil Evaluate and protect it with Vijil Dome.

## Setup

Clone this repository and create a new virtual environment:

```bash
python -m venv dome-tutorials
# Activate the virtual environment
# macOS/Linux:
source dome-tutorials/bin/activate
# Windows:
dome-tutorials\Scripts\activate
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root with the following environment variables:

```
GROQ_API_KEY=<Your Groq API key>
TAVILY_API_KEY=<Your Tavily Search API key>
VIJIL_API_KEY=<Your Vijil API key>
```

### API Key Setup

- **Groq API key**: Sign up at [groq.com](https://groq.com/)
- **Tavily API key**: Sign up at [tavily.com](https://www.tavily.com/)
- **Vijil API key**: Sign up at [evaluate.vijil.ai](https://evaluate.vijil.ai) and follow the [authentication instructions](https://docs.vijil.ai/setup.html#authentication-using-api-keys)

### Ngrok Setup (Free Plan Users Only)

If you're using Vijil's free plan, you'll need an Ngrok authorization token. Ngrok creates secure HTTPS endpoints for your agent. Premium Vijil subscribers can skip this step.

```
NGROK_AUTHTOKEN=<Your Ngrok authtoken>
```

Get your token at the [Ngrok dashboard](https://dashboard.ngrok.com/get-started/setup/python).

## Running the Scripts

### Evaluate Agent (Unprotected)

```bash
python -m evaluate_agent
```

### Evaluate Agent with Dome Protection

```bash
python -m evaluate_agent --protected=true
```

## Troubleshooting

- Ngrok Antivirus Warnings: Some Windows antivirus software may flag Ngrok as malicious. Ngrok is [safe and legitimate](https://ngrok.com/docs/faq/#is-ngrok-a-virus). When using Ngrok through our service, all traffic is guaranteed to come from Vijil's systems only.

- Rate Limiting: If you encounter rate limit errors, try reducing the request frequency or upgrade your Groq and Tavily accounts for higher limits.