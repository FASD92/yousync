import smtplib
from email.mime.text import MIMEText

def send_status_email(to_email: str, actor: str, new_status: str):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    from_email = "your_gmail@gmail.com"
    app_password = "your_app_password"  # 앱 비밀번호 필요

    subject = f"[YouSync] 요청하신 영상이 {new_status}되었습니다"
    body = f"""
    안녕하세요.

    배우 {actor}에 대한 영상 요청이 '{new_status}' 되었습니다.

    감사합니다.
    - YouSync 
    """

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(from_email, app_password)
        server.send_message(msg)
