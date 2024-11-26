import os
import fitz  # PyMuPDF for PDF parsing
import json
from transformers import pipeline
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer



class CoeusPDF:
    def __init__(self, pdf_path, save_dir, title=None):
        self.pdf_path = pdf_path
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize components
        self.text, self.structure = self.extract_pdf_text_structure(pdf_path)
        self.images = self.extract_pdf_images(pdf_path)
        self.nlp = spacy.load("en_core_web_sm")
        self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        self.sections = self.categorize_pdf_content(self.text, self.structure)
        self.problem_solution_map = self.build_problem_solution_map()

        # Save results as JSON
        self.save_output()

    def extract_pdf_text_structure(self, pdf_path):
        """Extract text and structural elements (headings, subheadings) from PDF."""
        doc = fitz.open(pdf_path)
        text = ""
        structure = []
        for page_num, page in enumerate(doc, start=1):
            text += page.get_text("text")
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block["type"] == 0:  # Text block
                    lines = block["lines"]
                    for line in lines:
                        structure.append({"text": line["spans"][0]["text"], "page": page_num})
        return text, structure

    def extract_pdf_images(self, pdf_path):
        """Extract images from PDF and associate them with pages."""
        doc = fitz.open(pdf_path)
        images = []
        for page_num, page in enumerate(doc, start=1):
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                img_path = os.path.join(self.save_dir, f"page_{page_num}_image_{xref}.png")
                with open(img_path, "wb") as img_file:
                    img_file.write(image_data)
                images.append({"path": img_path, "page": page_num})
        return images

    def categorize_pdf_content(self, text, structure):
        """Categorize extracted text using structural hints and keyword matching."""
        sections = []
        current_category = None
        current_text = ""
        keywords = {
            "installation": ["install", "setup", "configure", "mount"],
            "repair": ["repair", "fix", "troubleshoot", "issue", "fault"],
            "qa": ["question", "solution", "how to"],
            "maintenance": ["maintain", "service", "routine", "schedule"]
        }
        for item in structure:
            line = item["text"]
            for category, terms in keywords.items():
                if any(term in line.lower() for term in terms):
                    if current_category:
                        sections.append({"category": current_category, "text": current_text.strip()})
                    current_category = category
                    current_text = line + "\n"
                    break
            else:
                if current_category:
                    current_text += line + "\n"

        if current_category and current_text.strip():
            sections.append({"category": current_category, "text": current_text.strip()})
        return sections

    def build_problem_solution_map(self):
        """Build a structured mapping of problems to solutions."""
        problem_solution_map = []
        for section in self.sections:
            if section["category"] in ["repair", "qa"]:
                doc = self.nlp(section["text"])
                problems = []
                solutions = []
                for sent in doc.sents:
                    if any(keyword in sent.text.lower() for keyword in ["problem", "issue", "fault"]):
                        problems.append(sent.text.strip())
                    elif any(keyword in sent.text.lower() for keyword in ["solution", "fix", "steps"]):
                        solutions.append(sent.text.strip())

                for prob in problems:
                    matched_solution = solutions.pop(0) if solutions else "No solution provided."
                    problem_solution_map.append({"problem": prob, "solution": matched_solution})
        return problem_solution_map

    def save_output(self):
        """Save structured output as JSON for easier access."""
        output = {
            "text": self.text,
            "sections": self.sections,
            "images": self.images,
            "problem_solution_map": self.problem_solution_map
        }
        with open(os.path.join(self.save_dir, "output.json"), "w") as json_file:
            json.dump(output, json_file, indent=4)

    def answer_question(self, question):
        """Use a question-answering pipeline to answer a question."""
        context = " ".join([sec["text"] for sec in self.sections if sec["category"] in ["qa", "repair"]])
        if context:
            answer = self.qa_pipeline({"question": question, "context": context})
            return answer["answer"]
        return "Sorry, no answer found."

    def get_problem_solution(self, problem):
        """Retrieve solution for a specific problem."""
        for entry in self.problem_solution_map:
            if problem.lower() in entry["problem"].lower():
                return entry["solution"]
        return "No solution found for the problem."

    def get_image_for_section(self, category):
        """Retrieve the first image for the given category."""
        for image in self.images:
            if category in image["path"]:
                return image["path"]
        return None
