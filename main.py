from flask import Flask, request, abort

from linebot import (LineBotApi, WebhookHandler)

from linebot.exceptions import (InvalidSignatureError)

from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
)

import re

import AI

app = Flask(__name__)

token = ''
secret = ''
line_bot_api = LineBotApi(token)
handler = WebhookHandler(secret)


@app.route('/callback', methods=['POST'])
def callback():
  signature = request.headers['X-Line-Signature']
  body = request.get_data(as_text=True)
  app.logger.info("Request body: " + body)

  try:
    handler.handle(body, signature)
  except InvalidSignatureError:
    print(
        "Invalid signature. Please check your channel access token/channel secret."
    )
    abort(400)

  return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
  message = event.message.text

  reply_message = AI.AI_reply(message)

  line_bot_api.reply_message(event.reply_token,
                             TextSendMessage(text=reply_message))


app.run(host='0.0.0.0', port=8080)
