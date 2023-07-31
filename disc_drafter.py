import streamlit as st
import textwrap
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, OpenAIChat
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from datetime import datetime
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.docstore import InMemoryDocstore
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
st.title("Discovery Drafter")

#### OBJECTION INPUT SECTION ####
st.subheader("Objection Generator:")

st.sidebar.header("Objections")
objection = st.sidebar.multiselect("Type", ["Social Security", "Expert", "Req for Injury Description", "Reference to Disability", "Request for Records 10+ Years Prior","Boilerplate Overly Broad Medical Requests", "Request for Loan/Advance Info", 
                                          "Boilerplate Overly Broad Medical Requests", "Request for Records 10+ Years Prior", "Request for Loan/Advance Info", "Request for HIPAA", "Social Media", "Generic Witness Response"
                                            ])

if "Social Security" in objection:
    st.write("Plaintiff objects to providing his/her social security number for any purpose other than one authorized by Federal Law. Subject to and without waiving the foregoing objection...")
   
if "Expert" in objection:
    st.write("""
O.C.G.A. § 9-11-26(b)(4)(A)(i) states that a party may “require any other party to identify each person whom the other party expects to call as an expert witness at trial, to state the subject matter on which 
the expert is expected to testify, and to state the substance of the facts and opinions to which the expert is expected to testify and a summary of the grounds for each opinion.” 
However, O.C.G.A. § 9-11-26(b)(4)(A)(i) only applies to “experts whose knowledge of the facts and opinions held were acquired or developed in anticipation of litigation or for trial, and not to an expert witness 
who is in fact an actor or observer of the subject matter of the suit.” *Yang v. Smith*, 316 Ga. App. 458 (2012) (quoting *Stewart v. Odunkwe*, 273 Ga. App. 380 (2005)). 
The statute accordingly “does not apply to a physician whose knowledge and opinions arose from his own involvement in the plaintiff's medical care.” *Id.* (emphasis added). 
“The purpose of identifying witnesses [before trial] is to eliminate the possibility of surprise to each party.” *Stewart*, 273 Ga. App. at 381. 
“[W]here the complaining party cannot legitimately claim surprise, either because he knew of the existence of the witness or had equal means of knowing, it is not error to fail to invoke the sanctions of postponement, 
mistrial, barring the witness, etc.” *Kamensky v. Stacey*, 134 Ga. App. 530, 532 (1975). \n\n 
Plaintiff’s treating physicians may qualify as experts in the field of medicine in this matter. 
However, Plaintiff’s treating physicians are not experts as contemplated under O.C.G.A. § 9-11-26(b)(4)(A)(i). 
Regardless, to avoid any later confusion or surprise, Plaintiff hereby gives notice in this response that any and all of Plaintiff’s treating physicians may be called to testify at trial. 
\n\n All of Plaintiff’s treating physicians expected to testify at trial will be specifically identified in his/her portion of the Pre-Trial Order. 
All of Plaintiff’s treating providers will be further identified in Plaintiff’s discovery responses requesting that he/she identify his/her medical providers. 
All medical records and all documents in Plaintiff’s possession from each of these providers will be produced during the discovery period or as quickly as possible. 
\n\n Plaintiff’s treating physicians, if called to testify, are expected to testify about their examinations, diagnoses, treatment, prognosis, or interpretations of tests or examinations, including the basis therefor. 
They are expected to testify consistent with their medical records that will be produced to Defendant in discovery. 
They are further expected to testify regarding causation for Plaintiff’s injuries and the necessity and/or reasonableness of any medical treatment.
Plaintiff anticipates that his/her physicians will testify that he/she sustained physical injuries as a result of the incident that is the subject of this litigation. 
Plaintiff’s treating physicians’ opinions will be based on their respective training, education, and experience as well as their personal examinations, interactions with, and treatment of Plaintiff.
\n\n Plaintiff has not yet made a final determination as to which experts he/she will call to testify at trial. 
Plaintiff will supplement this response as required by law.
""")

if "Req for Injury Description" in objection:
    st.write("""
Plaintiff believes he/she sustained a multitude of traumatic injuries and refers Defendant to his/her medical records for more detailed explanation of the injuries he/she was diagnosed with. 
Plaintiff is not a medical practitioner and is therefore not qualified to define "each and every" injury he/she sustained. 
However, Plaintiff is able to state that he/she experienced pain in his neck pain, back pain, head pain, knee pain, wrist pain, headaches, and shoulder pain following the collision.
""")
    
if "Reference to Disability" in objection:
    st.write("""
Plaintiff objects to this request to the extent that the term “disabled” is confusing and ambiguous. 
Subject to the foregoing objection, Plaintiff does not contend that he/she was ever completely disabled as a result of his/her injuries. 
Plaintiff’s injuries were painful and significantly limited him/her at times, which could be construed as a type of disability. 
Plaintiff is not presently aware of any permanent disability ratings as a result of his/her injuries. 
However, Plaintiff defers to his/her treating doctors to discuss the level and type of any permanent disability if any such disability exists. 
Plaintiff reserves the right to supplement this response as discovery is ongoing.
""")

if "Social Media" in objection:
    st.write("""
Plaintiff objects to this request on the grounds that it is overly broad and seeks information that is neither relevant, nor reasonably calculated to lead to the discovery of admissible evidence. 
Courts routinely deny requests that amount to no more than fishing expeditions for discovery of private social media pages where the party fails to make a threshold showing to the relevancy of any information contained 
on an individual's private social 
networking websites, internet usage or email addresses. *See, e.g., Salvato v. Miley*, No. 5:12-CV-635-Oc-10PRL, 2013 U.S. Dist. LEXIS 81784, at *6-7 (M.D. Fla. 2013) 
(“The mere hope that Brown's private text-messages, e-mails, and electronic communication might include an admission against interest, without more, is not a sufficient reason to require Brown to provide 
Plaintiff open access to her private communications with third parties.”); *McCann v. Harleysville Ins. Co.*, 910 N.Y.S.2d 614, 78 A.D.3d 1524 (N.Y. App. Div. 2010) (“Although defendant specified the type of evidence sought, 
it failed to establish a factual predicate with respect to the relevancy of the evidence. Indeed, defendant essentially sought permission to conduct ‘a fishing expedition’ into plaintiff's Facebook account based on the mere hope of 
finding relevant evidence.”) (citations omitted); *see also Crispin v. Christian Audigier, Inc.*, 717 F. Supp. 2d 965 (C.D. Cal. 2010) (holding that Facebook and MySpace are Electronic Communication Services, and thus subject to the 
Stored Communications Act, 18 U.S.C. § 2701 et seq.).
\n\n Subject to and without waiving the foregoing objections, Plaintiff has not communicated any information about the subject incident or litigation on any online or social media website. 
""")

if "Request for HIPAA" in objection:
    st.write("""
Plaintiff objects and declines to provide any such authorization as he/she is under no legal obligation to do so. The decision to sign a HIPAA form is a voluntary decision that belongs to the patient. *Moreland v. Austin*, 284 GA. 730 (2008). 
According to the Georgia Supreme Court in *Allen v. Wright*, 282 Ga. 9 (2007), even the legislature cannot pass a law requiring a plaintiff to sign a medical authorization form unless it is more stringent than HIPAA and contains language that 
the person signing it can revoke it any time.
\n\n Defendant is able to use O.C.G.A. § 9-11-34 to acquire relevant and discoverable documents that he or she is legally entitled to. 
If it is necessary for Defendant to seek records from an out-of-state provider or entity, Plaintiff will not object to Defendant’s use of the Uniform Interstate Depositions and Discovery Act, 
O.C.G.A § 24-13-110 to obtain records related to the injuries Plaintiff received in the subject incident.
""")

if "Request for Loan/Advance Info" in objection:
    st.write("""
Plaintiff objects to this request as vague, overly broad, and objects to the extent that same seeks information which is not relevant or reasonably calculated to lead to the discovery of admissible evidence and therefore not within the scope of 
permissible discovery pursuant to the Georgia Civil Practice Act. Plaintiff further objects to this request to the extent that it seeks to embarrass, humiliate or unduly burden Plaintiff. This request further explicitly seeks what may be categorized as 
either collateral source information or wholly irrelevant information. 
\n\n Georgia’s collateral source rule generally prohibits a defendant from presenting evidence to the jury that a plaintiff previously received compensatory payments from other sources. *Andrews v. Ford Motor Co.*, 310 Ga. App. 449 (2011) 
(citing *Hoeflick v. Bradley*, 282 Ga. App. 123 (2006)). “A collateral source is generally a third party which has voluntarily provided a benefit through a bargained-for agreement, such as an insurer or as a gratuity.” *Olariu v. Marrerro*, 248 Ga. App. 824, 826 (2001).\n\n 
Collateral sources include, for example, payments by a plaintiff’s own insurer, but can include other sources as well, such as “beneficent bosses, or even helpful relatives.” *Andrews*, 210 Ga. App. at 124. 
“A tortfeasor cannot diminish the amount of his liability by pleading payments made to the plaintiff under the terms of a contract between the plaintiff and a third party who [is] not a joint tortfeasor.” *Broda v. Dziwura*, 286 Ga. 507, 509 (2010) 
(internal quotations and citations omitted)\n\n
At the outset, it is not even clear that the information sought in this request is collateral source information because the conferred benefit would technically not offset any of the tortfeasor’s liability or responsibility. 
Ironically, the information would be more relevant if it was collateral source information. In that event the information, though still inadmissible and overall irrelevant, would at least be marginally relevant to the matter-at-hand because it would 
touch on the issue of Plaintiff’s damages.\n\n
Assuming that the information sought is collateral source material, the request is objectionable to the extent that it exclusively seeks collateral source information without providing any justification as to how it is even loosely related 
to the subject litigation or the issues in dispute in the subject litigation – *i.e.* causation for Plaintiff’s injuries and the amount of Plaintiff’s damages. The fact that this request seeks strictly collateral source evidence that would certainly 
be inadmissible at trial immediately casts suspicion upon the request as a whole.\n\n
The request explicitly references “loan agreements,” which have no conceivable relevance to the underlying litigation. This overly broad request could theoretically extend to an agreement between Plaintiff and a 
family member for reimbursement of necessary living expense from an eventual resolution of Plaintiff’s claim. The fact that Plaintiff may or may not have engaged in a private, legitimate financial transactions for his own 
personal benefit is wholly irrelevant to the issues of this case. There is simply no rational argument as to how a private loan transaction would lead to relevant evidence in this case.\n\n
The financial status of a party in litigation is traditionally irrelevant and inadmissible at trial. *Warren v. Ballard*, 266 Ga. 408 (1996). Plaintiff would certainly not be permitted to present evidence at trial of
 his/her lack of financial means. Accordingly, discovery into Plaintiff’s financial means in this manner, particularly whether or not he/she needed or elected to take out loans during the pendency of this case, is improper and
   likely designed merely to frustrate, harass, embarrass and humiliate Plaintiff. If such discovery is permissible, Plaintiff should be permitted to conduct the same or similar discovery into Defendant’s finances and the finances 
   of the agents operating on behalf of Defendant, including Defendant’s insurer or those entities related to Defendant that have a financial stake in the outcome of this litigation. Plaintiff is skeptical, however, that Defendant’s counsel 
   would permit such discovery without first asserting vigorous objections.
""")

if "Request for Records 10+ Years Prior" in objection:
    st.write("""
Plaintiff objects to this request because it seeks information and records older than ten (10) years and is therefore unreasonable and unduly burdensome.
Medical facilities in this State are only required to retain medical records for ten (10) years, which is an impliedly reasonable period of time. O.C.G.A. § 31-33-2(a)(1)(A) (“A provider … shall retain such item for a period of not less than ten 
years from the date such item was created.”). Defendant’s request seeks information and records older than ten (10) years and is therefore *de facto* unreasonable and unduly burdensome. 
""")

if "Boilerplate Overly Broad Medical Requests" in objection:
    st.write("""
Plaintiff objects to this request to the extent that it requests information outside the scope of discovery. 
Plaintiff has only placed his/her health relevant to the subject incident in dispute by filing the subject lawsuit. 
Discovery regarding *all* of Plaintiff’s medical history, even those issues not related to the injuries Plaintiff is currently claiming in the subject litigation, is therefore not appropriate and outside the scope of valid discovery. 
\n\n Subject to and without waiving the foregoing objection(s),...
""")

if "Generic Witness Response" in objection:
    st.write("""
Plaintiff directs Defendant to the Georgia Uniform Motor Vehicle Accident Report for the subject incident and any entities identified therein. 
Plaintiff is unaware of any eyewitnesses to the subject collision aside from those mentioned in the police report for the subject collision. 
Plaintiff believes that himself/herself, Defendant, the investigating police officer, Plaintiff’s medical providers, and Plaintiff’s friends and family may possess information relevant to the subject occurrence, 
any issue of liability, or Plaintiff’s damages and causation for those damages. Plaintiff reserves the right to amend and supplement this response as more information becomes available through discovery.
""")


#### QUESTIONNAIRE QUERY SECTION ####

st.subheader("Litigation Questionnaire Query:")

import os
load_dotenv()
os.getenv("OPENAI_API_KEY")

#### QUERY SECTION ####

# upload file
pdf_docs = st.file_uploader("Upload your Litigation Questionnaire", type="pdf", accept_multiple_files=True)

def get_pdf_text (pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

    # load the pdf here
    # then assign it to pdf_docs
if pdf_docs:
    st.write("PDF was loaded successfully")
else:
    st.write("No document provided.")

# extract the text
if pdf_docs is not None:
    raw_text = get_pdf_text(pdf_docs)

# get the chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

text_chunks = get_text_chunks(raw_text)

# set up the vector store
memory = ConversationBufferMemory(
    memory_key="chat history",
    return_messages=True
)

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorestore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorestore

if len(pdf_docs) > 0:
    vectorestore = get_vectorstore(text_chunks)

temperature = st.sidebar.number_input("Query Response Creativity (lower number, less creative)", min_value=0.0, max_value=1.0, step=0.1, value=0.2)
num_resp = st. sidebar.number_input("Number of Outputs (recommend atleast 2 for complex questions)",
                                    min_value=1, max_value=5, step=1, value=1)
# user input
user_question = st.text_input("Ask a question:")
if st.button("Query Litigation Questionnaire"):
    with st.spinner("Processing..."): 
        docs = vectorestore.similarity_search(user_question)
        
        llm = OpenAIChat(temperature=temperature, model="gpt-3.5-turbo")
        chain = load_qa_chain(llm, chain_type="stuff")
        
        for i in range(num_resp):
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, 
                                    question=user_question, 
                                    memory=memory,
                                    return_source_documents=True)
                print(cb)

            st.write(response)
