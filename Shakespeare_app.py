import streamlit as st
import time
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch
# from streamlit_chat import message as st_message


html_temp= """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:centre;"> Shakespeare-AI.</h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

st.markdown(
   
    """
    This is an AI tool. This AI chatbot is able to talk to you like Shakespeare.
"""
)
output_path = "SurendraKumarDhaka/output" 
# Load the trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(output_path)
model = GPT2LMHeadModel.from_pretrained(output_path)

input_text = st.text_area("Type your text..")
st.button("Submit")
input_ids = tokenizer.encode(input_text, return_tensors="pt")
# Generate text using the model
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
st.write(generated_text) 
st.text("Thanks for using")
            
if st.button("About"):
        st.text("Created by Surendra Kumar")
## footer
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="solid",
        border_width=px(0.5)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )
    st.markdown(style,unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "©️ surendraKumar",
        br(),
        link("https://www.linkedin.com/in/surendra-kumar-51802022b", image('https://icons.getbootstrap.com/assets/icons/linkedin.svg') ),
        br(),
        link("https://www.instagram.com/im_surendra_dhaka/",image('https://icons.getbootstrap.com/assets/icons/instagram.svg')),
    ]
    layout(*myargs)

if __name__ == "__main__":
    footer()