import sys
import pathlib
import csv
from time import sleep

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from shell.process import Process
from model.GPT import GPT
import tkinter as tk
from typing import Self


BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

GPT_LEFT_TAB = "#202123" 
GPT_BOT_ANSWER = "#343541" 
GPT_USER_INPUT = "#444654" 
GPT_TEXT_BOX = "#40414F" 


FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


class HMI:
    '''
        This class is the GUI of the chatbot.
    '''

    def __init__(self: Self, model: GPT, proc: Process) -> None:
        self.__model = model
        self.__proc = proc
        self.__answer = ''
        self.__gui = tk.Tk()
        self.initialize()

    @property
    def model(self: Self):
        '''
            This property returns the model.
        '''
        return self.__model

    @property
    def gui(self: Self):
        '''
            This property returns the GUI.
        '''
        return self.__gui

    def initialize(self: Self):
        '''
            This method initializes the GUI.
        '''
        self.model.fit()
        self.gui.title("Chat")
        self.gui.resizable(width=True, height=True)
        self.gui.configure(width=470, height=550, bg=BG_COLOR)

        # head label
        head_label = tk.Label(self.gui, bg=BG_COLOR, fg=TEXT_COLOR,
                              text="F.R.I.D.A.Y.", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = tk.Label(self.gui, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_area = tk.Text(self.gui, width=20, height=2,
                                 bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_area.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_area.configure(cursor="arrow", state=tk.DISABLED)

        # scroll bar
        scrollbar = tk.Scrollbar(self.text_area)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_area.yview)

        # bottom label
        bottom_label = tk.Label(self.gui, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message box
        bottom_label2 = tk.Label(bottom_label, bg=BG_COLOR, height=80)
        bottom_label2.place(relwidth=0.74 + 0.24,
                            relheight=0.06, rely=0.008, relx=0.011)

        # message entry box
        self.text_box = tk.Entry(bottom_label2, bg=BG_COLOR, fg=TEXT_COLOR,
                                 font=FONT, relief=tk.FLAT, highlightthickness=0, borderwidth=0)
        self.text_box.place(relwidth=0.9, relheight=1, rely=0, relx=0)
        self.text_box.focus()
        self.text_box.bind("<Return>", self.send)

        # send button
        self.__button = tk.Button(bottom_label2, text="Send", font=FONT_BOLD, width=16, bg=BG_GRAY,
                                  command=self.send, relief=tk.FLAT)
        self.__button.place(relx=0.9, rely=0.2, relheight=0.6, relwidth=0.1)

    def run(self) -> None:
        '''
            This method runs the GUI.
        '''
        self.gui.mainloop()

    def send(self: Self, event: tk.Event|None=None) -> None:
        '''
            This method sends a message to the chatbot.

            Args:
                event (tk.Event): The event that triggered the method.
        '''
        message = self.text_box.get()
        if not message:
            return
        self.__user_input = message
        self.text_box.delete(0, tk.END)
        
        self.__send(message, event)

    def regenerate_response(self: Self, event: tk.Event|None=None) -> None:
        '''
            This method is called when the regenerate button is hit, resending the previous prompt to the bot.

            Args:
                event (tk.Event): The event that triggered the method.
        '''
        self.__send(self.__user_input, event)

    def __send(self: Self, message: str, event: tk.Event|None=None) -> None:
        '''

        '''
        while (answer := self.answer_to(message)) == self.__answer:
            continue
        self.__answer = self.answer_to(message)
        self.text_area.configure(state=tk.NORMAL)
        self.text_area.insert(tk.END, "You: " + message + '\n\n')
        self.text_area.insert(tk.END, "F.R.I.D.A.Y: ")
        self.text_area.insert(tk.END, self.__answer + ("\n" * 2))
        self.text_area.configure(state=tk.DISABLED)

    def approved(self: Self, event: tk.Event|None=None) -> None:
        '''
            This method is called when the like button is hit, triggering a change in the training dataset of the bot.

            Args:
                event (tk.Event): The event that triggered the method.
        '''
        with (pathlib.Path(__file__).parent.parent.parent.parent / 'datasets' / 'approved.csv').open('a') as dataset:
            csv.writer(dataset).writerow([self.__user_input, self.__answer])
        self.model.check_db_change()

    def answer_to(self: Self, message: str) -> str:
        '''
            This method returns the answer of the chatbot to a message.

            Args:
                message (str): The message to which the chatbot will answer.
        '''
        return self.model(message)
