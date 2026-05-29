"""
Email Agent - Main Entry Point
Run this file to start the interactive email agent CLI.
"""

import sys
from agent.email_agent import EmailAgent
from config.settings import Settings


def main():
    print("\n" + "="*60)
    print("  📧  AI Email Agent  |  Powered by OpenAI + Gmail")
    print("="*60)
    print("\nInitializing agent...\n")

    try:
        settings = Settings()
        agent = EmailAgent(settings)
        agent.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Check your .env file and credentials. See README.md for setup.")
        sys.exit(1)


if __name__ == "__main__":
    main()
