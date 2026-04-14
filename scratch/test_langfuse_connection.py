import os
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

def test_langfuse():
    print(f"Checking Langfuse Host: {os.getenv('LANGFUSE_HOST')}")
    lf = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    
    try:
        # Try to send a simple trace
        trace = lf.trace(name="connection-test")
        print(f"Trace created: {trace.id}")
        
        # Add a span
        span = trace.span(name="ping")
        span.end(output="pong")
        
        # Flush to ensure it's sent
        lf.flush()
        print("✅ Trace sent and flushed.")
    except Exception as e:
        print(f"❌ Langfuse Error: {e}")

if __name__ == "__main__":
    test_langfuse()
