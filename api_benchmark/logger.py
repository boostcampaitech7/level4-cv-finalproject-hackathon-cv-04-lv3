import time

def log_time(LOG_FILE, message):
    log_message = f"{message}"
    
    with open(LOG_FILE, "a") as log_file:
        log_file.write(log_message + "\n")
    
    print(log_message)