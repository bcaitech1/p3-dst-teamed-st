### Task-oriented dialogue vs Open-domain dialogue(Chit-chat)

Open-domain dialogue 는 자연스러움이 중요하지만, Task-oriented는 task의 수행성공여부가 중요함.

### TaskOrientedDialogue
Pre-defined scenario 에서 Task schema 를 수행
- Task-schema  
  유저는 자신의 상황을 Informable slot으로 설명하고 서버에 Request
  서버는 Informable slot과 Request를 참고하여, Requestable slot에서 정보를 제공한다.
  이 때, Informable slot 들을 추적하는 것이 Dialogue state tracking