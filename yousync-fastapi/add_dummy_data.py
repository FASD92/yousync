from database import SessionLocal, engine
from models import Base, Script

Base.metadata.create_all(bind=engine)

def add_dummy_data():
    db = SessionLocal()
    try:
        all_data = [
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 0.0,
                "end_time": 4.0,
                "script": "Berni everyone's burning with desire",
                "translation": "여러분 모두 욕망으로 불타고 있습니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 4.0,
                "end_time": 7.0,
                "script": "we're burning with a fire caused by what",
                "translation": "우리는 무엇이 일으킨 불꽃으로 타오르고 있습니까?",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 10.0,
                "end_time": 15.0,
                "script": "greed anger ignorance except he taught",
                "translation": "탐욕, 분노, 무명을 말하지만, 부처님은",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 15.0,
                "end_time": 18.0,
                "script": "us that we can fix this we can turn them",
                "translation": "우리가 이를 고칠 수 있고, 그것들을",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 18.0,
                "end_time": 25.0,
                "script": "around and greed becomes generosity",
                "translation": "바꾸어 탐욕이 보시(관대함)가 되게 할 수 있다고 가르치셨습니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 25.0,
                "end_time": 31.0,
                "script": "anger becomes compassion an ignorance",
                "translation": "분노는 자비가 되고, 무명은",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 31.0,
                "end_time": 34.0,
                "script": "becomes wisdom",
                "translation": "지혜가 된다고 하셨습니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 34.0,
                "end_time": 37.0,
                "script": "[Music]",
                "translation": "[음악]",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 37.0,
                "end_time": 39.0,
                "script": "there are miracles around us all the",
                "translation": "우리 주위에는 언제나 기적이 있습니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 39.0,
                "end_time": 43.0,
                "script": "time he said the fact that we are here",
                "translation": "그는 우리가 이 자리에 함께 있는 것 자체가",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 43.0,
                "end_time": 48.0,
                "script": "together in this room is a miracle isn't",
                "translation": "기적이라고 말씀하셨습니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 48.0,
                "end_time": 51.0,
                "script": "the Buddha kind of a hypocrite I mean",
                "translation": "근데 부처님이 위선자 같지 않나요?",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 51.0,
                "end_time": 53.0,
                "script": "sure he gave up all worldly goods his",
                "translation": "분명히 그는 세속의 모든 것을 내려놓았고,",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 53.0,
                "end_time": 55.0,
                "script": "father's money told us all to live a",
                "translation": "아버지의 재산도 버리고 금욕적인 삶을 살라고 하셨죠",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 55.0,
                "end_time": 58.0,
                "script": "life of restraint and then he ended up",
                "translation": "그런데 결국 그는",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 58.0,
                "end_time": 61.0,
                "script": "like super fat like crazy fat like come",
                "translation": "굉장히 뚱뚱해졌는데, 말도 안 되게 말이죠",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 61.0,
                "end_time": 64.0,
                "script": "on dude you're talking about the statues",
                "translation": "야, 너 차이나타운에 있는 동상 이야기하는 거지?",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 64.0,
                "end_time": 66.0,
                "script": "you see in Chinatown right yeah the",
                "translation": "그 동상들 말이야. 맞아,",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 66.0,
                "end_time": 68.0,
                "script": "Buddha depicted in those statutes he's",
                "translation": "그 동상에 묘사된 부처는",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 68.0,
                "end_time": 72.0,
                "script": "Asian right right so it's not the Buddha",
                "translation": "아시아인이지? 그래서 그건 진짜 부처님이 아니야",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 72.0,
                "end_time": 75.0,
                "script": "okay wait what do you mean well think",
                "translation": "잠깐, 무슨 말이야? 잘 생각해 봐",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 75.0,
                "end_time": 76.0,
                "script": "about it",
                "translation": "생각해봐",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 76.0,
                "end_time": 79.0,
                "script": "Siddhartha Gautama was an Indian from",
                "translation": "싯다르타 고타마는 네팔 출신의 인도인이었어요",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 79.0,
                "end_time": 82.0,
                "script": "Nepal a Chinese monk came to India",
                "translation": "중국인 승려가 인도로 와서",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 82.0,
                "end_time": 85.0,
                "script": "became a Buddhist went home and started",
                "translation": "불교를 받아들인 뒤 고국으로 돌아가",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 85.0,
                "end_time": 87.0,
                "script": "spreading the religion the Chinese mixed",
                "translation": "종교를 전파했어요. 그 과정에서 중국은",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 87.0,
                "end_time": 90.0,
                "script": "in Taoism they're monks introduced the",
                "translation": "도교를 섞고, 승려들이 머리 민 것과",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 90.0,
                "end_time": 94.0,
                "script": "head shaving the robes the fat happy",
                "translation": "승복 스타일, 뚱뚱하고 행복한",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 94.0,
                "end_time": 98.0,
                "script": "Buddha named Hotei the actual Buddha was",
                "translation": "부처 ‘호테이’ 스타일을 도입했죠. 실제 부처님은",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 98.0,
                "end_time": 100.0,
                "script": "a regular man normal body full head of",
                "translation": "보통 사람이었고, 머리도 풍성했어요",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 100.0,
                "end_time": 104.0,
                "script": "hair but he was a God no never claimed",
                "translation": "하지만 그가 신이었다고는 절대 주장하지 않았습니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 104.0,
                "end_time": 107.0,
                "script": "to be just a man he thought long and",
                "translation": "그는 그저 인간으로서 인간의 고통에 대해",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 107.0,
                "end_time": 109.0,
                "script": "hard about the human condition and",
                "translation": "오랫동안 고뇌했고,",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 109.0,
                "end_time": 115.0,
                "script": "achieved enlightenment wait Jesus only without the long blonde hair",
                "translation": "깨달음을 얻었어요. 예수님과 비슷하죠. 단, 긴 금발 머리는 없었지만요",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 115.0,
                "end_time": 119.0,
                "script": "but Jesus wasn't just a man no and he",
                "translation": "하지만 예수님은 단순한 인간이 아니었습니다. 그는",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 119.0,
                "end_time": 121.0,
                "script": "claimed to be the Son of God died and",
                "translation": "스스로를 하나님의 아들이라 주장했고, 죽었다가",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 121.0,
                "end_time": 125.0,
                "script": "was resurrected so we know he was divine",
                "translation": "부활했으니 그의 신성을 알 수 있죠",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 125.0,
                "end_time": 129.0,
                "script": "you know or you have faith you can only",
                "translation": "그렇죠? 아니면 믿음을 가져야만",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 129.0,
                "end_time": 132.0,
                "script": "know what you can prove God is beyond",
                "translation": "알 수 있어요. 하나님은 논리만으로 증명할 수 없는 존재니까요",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 132.0,
                "end_time": 135.0,
                "script": "proof logical reason you can believe in",
                "translation": "이성이나 논리로는 알 수 없기에 믿음으로 하나님이나",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 135.0,
                "end_time": 140.0,
                "script": "God or Jesus through faith but that's",
                "translation": "예수를 믿을 수 있지만, 그건 다른 이야기예요",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 140.0,
                "end_time": 143.0,
                "script": "different from no why no Jesus existed",
                "translation": "아니에요. 예수님은 실제로 존재했다는",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 143.0,
                "end_time": 145.0,
                "script": "there's historical evidence of that and",
                "translation": "역사적 증거가 있으니까요",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 145.0,
                "end_time": 149.0,
                "script": "I have faith that you have divine quiner",
                "translation": "나는 당신에게 신성이 있다고 믿습니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 149.0,
                "end_time": 150.0,
                "script": "said faith is a divine act",
                "translation": "키에르케고르는 ‘믿음은 신성한 행위’라고 말했습니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 150.0,
                "end_time": 153.0,
                "script": "supernaturally bestow ask any shall",
                "translation": "초자연적으로 주어집니다. 구하면 누구든",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 153.0,
                "end_time": 155.0,
                "script": "receive Kierkegaard on the other hand",
                "translation": "받습니다. 반면 키에르케고르는",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 155.0,
                "end_time": 158.0,
                "script": "said that we must leap to faith it is an",
                "translation": "‘믿음으로 도약해야 한다’고 말했죠",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 158.0,
                "end_time": 161.0,
                "script": "act that you must choose to perform",
                "translation": "그것은 당신이 선택해서 행해야 하는 행위입니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 161.0,
                "end_time": 164.0,
                "script": "Kierkegaard was more demanding I can he",
                "translation": "키에르케고르는 더 엄격했어요",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 164.0,
                "end_time": 165.0,
                "script": "be sick of people sitting around talking",
                "translation": "사람들이 종일 종교 얘기만 하고도",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 165.0,
                "end_time": 167.0,
                "script": "about religion all day and not doing anything about it",
                "translation": "아무 행동도 안 하는 것에 지쳤을 거죠",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 167.0,
                "end_time": 170.0,
                "script": "I hope he didn't mean this class though",
                "translation": "하지만 이 수업만큼은 아니었길 바라네요",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 170.0,
                "end_time": 172.0,
                "script": "but uh weren't we supposed to be talking",
                "translation": "그런데 저희 부처님에 대해 이야기하기로 하지 않았나요?",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 172.0,
                "end_time": 175.0,
                "script": "about Buddha they had a lot in common",
                "translation": "부처님에 대해요? 그들은 많은 공통점이 있죠",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 175.0,
                "end_time": 180.0,
                "script": "Jesus and Buddha how so well there's a",
                "translation": "예수님과 부처님, 어떻게요? 그들의",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 180.0,
                "end_time": 182.0,
                "script": "lot of overlap in their philosophies",
                "translation": "철학에는 많은 겹침이 있습니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 182.0,
                "end_time": 184.0,
                "script": "they both taught the golden rule that we",
                "translation": "그들은 모두 ‘황금률’, 즉",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 184.0,
                "end_time": 186.0,
                "script": "should be charitable and not judge",
                "translation": "남을 판단하지 말고 자비를 베풀라고 가르쳤습니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 186.0,
                "end_time": 189.0,
                "script": "others but the Buddha said you shouldn't",
                "translation": "하지만 부처님은 ‘경전에 쓰여 있다고 해서",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 189.0,
                "end_time": 190.0,
                "script": "believe in something just because it's",
                "translation": "무턱대고 믿지 말라’고 하셨죠",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 190.0,
                "end_time": 192.0,
                "script": "written in Scripture the whole premise",
                "translation": "반면 예수님은 믿음으로 받아들이라 하니",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 192.0,
                "end_time": 194.0,
                "script": "with Jesus is to accept on faith that's",
                "translation": "상당히 큰 차이라고 할 수 있죠",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 194.0,
                "end_time": 196.0,
                "script": "a pretty huge difference also the",
                "translation": "또한",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 196.0,
                "end_time": 198.0,
                "script": "purpose of suffering Jesus suffered for",
                "translation": "고통의 목적은 예수님은 우리를 위해",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 198.0,
                "end_time": 200.0,
                "script": "us while the Buddhist whole thing was to",
                "translation": "고통을 겪으셨지만, 불교의 핵심은",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 200.0,
                "end_time": 202.0,
                "script": "eliminate suffering eliminate desire",
                "translation": "고통을 없애기 위해 욕망을 제거하라는 것입니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 202.0,
                "end_time": 204.0,
                "script": "which is what causes suffering I know I",
                "translation": "욕망이 고통의 원인이니까요. 알죠?",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 204.0,
                "end_time": 209.0,
                "script": "listened I was listening okay next time",
                "translation": "들었어요, 듣고 있어요. 좋아요. 다음번엔",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 209.0,
                "end_time": 212.0,
                "script": "we will cover the path to Nirvana and we",
                "translation": "열반으로 가는 길을 다루고",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 212.0,
                "end_time": 214.0,
                "script": "will wrap this semester up with Jainism",
                "translation": "이번 학기를 자이나교로 마무리할 예정입니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 214.0,
                "end_time": 217.0,
                "script": "it's like Buddhism the less fun Jain",
                "translation": "자이나교는 불교와 비슷하지만 덜 재미있어요",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 217.0,
                "end_time": 219.0,
                "script": "women have to be reborn as men to",
                "translation": "여성이 깨달음을 얻으려면 남성으로 환생해야 합니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 219.0,
                "end_time": 222.0,
                "script": "achieve enlightenment we all know it",
                "translation": "깨달음을 얻기 위해서 말이죠",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 222.0,
                "end_time": 223.0,
                "script": "should be the other way around",
                "translation": "거꾸로 되어야 한다고 모두들 말하죠",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            },
            {
                "movie_id": 1,
                "actor_id": 1,
                "start_time": 223.0,
                "end_time": 224.0,
                "script": "thank you",
                "translation": "감사합니다",
                "actor_pitch_values": [0],
                "background_audio_url": ""
            }
        ]

        for data in all_data:
            script = Script(**data)
            db.add(script)

        db.commit()
        print("✅ 모든 더미 데이터가 성공적으로 추가되었습니다!")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    add_dummy_data()
