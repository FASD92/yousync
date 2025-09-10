import boto3
import json
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SQSService:
    """SQS 메시지 전송 서비스"""
    
    def __init__(self):
        self.sqs_client = boto3.client(
            'sqs',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'ap-northeast-2')
        )
        self.queue_url = os.getenv('SQS_QUEUE_URL')
        
        if not self.queue_url:
            logger.warning("SQS_QUEUE_URL 환경 변수가 설정되지 않았습니다.")
    
    def send_analysis_message(
        self, 
        job_id: str, 
        s3_audio_url: str, 
        token_id: str,
        webhook_url: str,
        token_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        분석 작업 메시지를 SQS에 전송
        
        Args:
            job_id: 작업 ID
            s3_audio_url: S3 오디오 파일 URL
            token_id: 토큰 ID
            webhook_url: 결과 수신용 웹훅 URL
            token_info: 토큰 관련 추가 정보
            
        Returns:
            SQS 메시지 ID 또는 None (실패시)
        """
        if not self.queue_url:
            logger.error("SQS_QUEUE_URL이 설정되지 않아 메시지를 전송할 수 없습니다.")
            return None
        
        try:
            # SQS 메시지 구성
            message_body = {
                "job_id": job_id,
                "s3_audio_url": s3_audio_url,
                "video_id": token_id,  # 기존 분석 서버 호환성을 위해 video_id 사용
                "webhook_url": webhook_url,
                "s3_textgrid_url": f"s3://testgrid-pitch-bgvoice-yousync/{token_info.get('s3_textgrid_url')}" if token_info.get('s3_textgrid_url') else None,
                "s3_pitch_url": f"s3://testgrid-pitch-bgvoice-yousync/{token_info.get('s3_pitch_url')}" if token_info.get('s3_pitch_url') else None,
                "timestamp": datetime.utcnow().isoformat(),
                "source": "fastapi_backend"
            }
            
            # None 값 제거
            message_body = {k: v for k, v in message_body.items() if v is not None}
            
            # SQS에 메시지 전송
            response = self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(message_body, ensure_ascii=False),
                MessageAttributes={
                    'job_type': {
                        'StringValue': 'audio_analysis',
                        'DataType': 'String'
                    },
                    'priority': {
                        'StringValue': 'normal',
                        'DataType': 'String'
                    },
                    'job_id': {
                        'StringValue': job_id,
                        'DataType': 'String'
                    }
                }
            )
            
            message_id = response.get('MessageId')
            logger.info(f"[SQS 전송 성공] job_id={job_id}, message_id={message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"[SQS 전송 실패] job_id={job_id}, error={str(e)}")
            return None
    
    def get_queue_attributes(self) -> Optional[Dict[str, Any]]:
        """큐 상태 정보 조회"""
        if not self.queue_url:
            return None
            
        try:
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['ApproximateNumberOfMessages', 'ApproximateNumberOfMessagesNotVisible']
            )
            return response.get('Attributes', {})
        except Exception as e:
            logger.error(f"큐 상태 조회 실패: {str(e)}")
            return None

# 싱글톤 인스턴스
sqs_service = SQSService()
