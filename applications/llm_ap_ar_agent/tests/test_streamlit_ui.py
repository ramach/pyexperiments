# tests/test_streamlit_ui.py
import unittest
from streamlit.testing.v1 import AppTest

class TestStreamlitUI(unittest.TestCase):

    def setUp(self):
        self.app = AppTest.from_file("ui/streamlit_app.py")

    def test_ui_renders_chat_input(self):
        self.app.run()
        chat_input = self.app.chat_input["Ask something about AP, AR, or vendors..."]
        self.assertTrue(chat_input.exists())

    def test_send_message(self):
        self.app.run()
        chat_input = self.app.chat_input["Ask something about AP, AR, or vendors..."]
        chat_input.set_value("List overdue payments")
        self.app.run(timeout=5)
        output = self.app.chat_message("assistant")
        self.assertTrue(output.exists())

if __name__ == '__main__':
    unittest.main()
