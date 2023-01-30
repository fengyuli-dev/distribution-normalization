import requests
import traceback
from pathlib import Path


def progress_alerts(original_function=None, *, func_name="clip-metric"):
    def _progress_alerts(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                if post_slack_message(f"Error in training {func_name}"):
                    stack = traceback.format_exc()
                    post_slack_message(stack)
                    post_slack_message("-" * 80)
                else: 
                    traceback.print_exc()
                exit()
            post_slack_message(f"{func_name} finishing running!")
        return wrapper 

    if original_function:
        return _progress_alerts(original_function)
    
    return _progress_alerts
        

def post_slack_message(msg):
    user = str(Path.home())
    if "jlr429" in user:
        msg = msg.replace("\"", "-")
        msg = msg.replace("\'", "-")
        payload = '{"text":"%s"}' % msg
        requests.post("https://hooks.slack.com/services/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", payload)
        return True
    else:
        return False
    