import os
import fitz  # PyMuPDF for PDF parsing
import json
import re
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import transformers
from sentence_transformers import SentenceTransformer, util
import spacy
import camelot
import shutil
import pytesseract
from PIL import Image
import requests
from io import BytesIO
import threading
from spacy.cli import download
import torch.multiprocessing as mp
import argparse
import logging
from .keywords import keyword_patterns as kp


class CoeusPdfProcessor:
    def __init__(self, pdf_path=None, save_dir=None, title=None, to_dataset=False, use_spacy=False):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "1"
        mp.set_start_method('spawn', force=True)
        self.logger = logging.getLogger(__name__)


        if pdf_path and  not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        self.pdf_path = pdf_path
        self.save_dir = os.path.join(save_dir, title) if title else save_dir
        self.to_dataset = to_dataset
        self.use_spacy = use_spacy  # Toggle between SpaCy and Hugging Face NLP
        os.makedirs(self.save_dir, exist_ok=True)

        # Load NLP model
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "deepset/roberta-base-squad2")
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                "deepset/roberta-base-squad2")
            self.qa_pipeline = pipeline(
                "question-answering", model=self.model, tokenizer=self.tokenizer)
            self.summarizer = pipeline(
                "summarization", model="facebook/bart-large-cnn")
            
            self.nlp = SentenceTransformer("all-MiniLM-L6-v2")

        # Extract data
        if pdf_path:
            self.structure = self.extract_pdf_text_structure(
                pdf_path) if pdf_path else ("", [])
            
        
        #     self.sections = self.categorize_pdf_content(self.structure)
        #     print(self.sections)
        #     self.images = self.extract_pdf_images(pdf_path) if pdf_path else []
        #     self.problem_solution_map = self.build_problem_solution_map()

        # # Generate dataset if required
        #     if to_dataset:
        #         self.generate_and_save_dataset()

    def extract_pdf_text_structure(self, pdf_path):
        """Extract hierarchical text structure from the PDF."""
        try:
            doc = fitz.open(pdf_path)
        except fitz.FileDataError as e:
            self.logger.error(f"Failed to open PDF file: {pdf_path} with {e}")
            raise ValueError(f"Failed to open PDF file: {pdf_path}") from e
        
        #print(doc.metadata, "-------------------1\n")
        print(doc.get_toc(False), "---------------2 \n")


        # content_structure = []
        # current_heading = None

        # for page_num, page in enumerate(doc, start=1):
        #     blocks = page.get_text("dict")["blocks"]
        #     for block in blocks:
        #         if block["type"] == 0:  # Text block
        #             lines = block["lines"]
        #             for line in lines:
        #                 spans = line["spans"]
        #                 for span in spans:
        #                     text = span["text"].strip()
        #                     font_size = span.get("size", 0)

        #                     if font_size > 12:  # Assume larger font sizes are headings
        #                         if current_heading:
        #                             content_structure.append(current_heading)
        #                         current_heading = {"heading": text, "content": [], "page": page_num}
        #                     elif current_heading:
        #                         current_heading["content"].append(text)
        
        # if current_heading:
        #     content_structure.append(current_heading)
        
        # return content_structure



    def extract_pdf_images(self, pdf_path):
        """Extract images from the PDF and perform OCR."""
        try:
            doc = fitz.open(pdf_path)
        except fitz.FileDataError as e:
            self.logger.error(f"Failed to open PDF file: {pdf_path} with {e}")
            raise ValueError(f"Failed to open PDF file: {pdf_path}") from e
        images = []
        for page_num, page in enumerate(doc, start=1):
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]

                # Save image locally
                img_path = os.path.join(self.save_dir, f"page_{
                                        page_num}_image_{xref}.png")
                with open(img_path, "wb") as img_file:
                    img_file.write(image_data)

                # OCR the image
                try:
                    image = Image.open(BytesIO(image_data))
                    ocr_text = pytesseract.image_to_string(image)
                except Exception as e:
                    ocr_text = f"OCR failed for {img_path}: {e}"

                images.append({"path": img_path, "page": page_num,
                              "ocr_text": ocr_text.strip()})
        return images

    def associate_images_with_text(images, text_structure):
        """Link images to their closest text content."""
        associations = []
        for img in images:
            page_num = img["page"]
            text_on_page = [t for t in text_structure if t["page"] == page_num]
            if text_on_page:
                associations.append({"image": img, "text": " ".join(t["text"] for t in text_on_page)})
        return associations
    
    def generate_image_captions(self):
        """Generate captions for images using OCR and NLP."""
        captions = []
        for img in self.images:
            if img["ocr_text"]:
                caption = self.summarize_section(img["ocr_text"])
                captions.append({"image_path": img["path"], "caption": caption})
        return captions

    def extract_tables_from_pdf(pdf_path):
        """Extract tables from PDF using Camelot."""
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        table_data = [table.df.to_dict(orient="records") for table in tables]
        return table_data

    def categorize_pdf_content(self, structure):
        """Categorize text using headings and keywords with regex support."""
        sections = []
        current_category = None
        current_text = ""
        
        # Define regex patterns for keywords
        keyword_patterns = kp
        # Compile regex patterns for performance
        compiled_patterns = {category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
                            for category, patterns in keyword_patterns.items()}
        #print(compiled_patterns)
        
        last_item=None
        for item in structure:
            last_item = item
            line = item["text"]
            matched_category = None

            # Check each category's patterns
            for category, patterns in compiled_patterns.items():
                if any(pattern.search(line) for pattern in patterns):
                    print(f" matched category {category}")
                    matched_category = category
                    break

            if matched_category:
                if current_category:  # If there's an existing category, save it
                    sections.append({
                        "heading": item["heading"],
                        "category": current_category,
                        "text": current_text.strip()
                    })
                current_category = matched_category
                current_text = line + "\n"
            else:
                if current_category:
                    current_text += line + "\n"

        # Append the last collected text to its category
        if current_category and current_text.strip():
            sections.append({"category": current_category, "text": current_text.strip(), "heading": last_item["heading"]})
        return sections


    def build_problem_solution_map(self):
        """Map problems to solutions."""
        problem_solution_map = []
        for section in self.sections:
            if section["category"] == "qa":
                if self.use_spacy:
                    doc = self.nlp(section["text"])
                    problems = [
                        sent.text for sent in doc.sents if "problem" in sent.text.lower()]
                    solutions = [
                        sent.text for sent in doc.sents if "solution" in sent.text.lower()]
                    for prob, sol in zip(problems, solutions):
                        problem_solution_map.append(
                        {"problem": prob, "solution": sol})
                else:
                    problems = [sent for sent in section["text"].split(".") if "problem" in sent.lower()]
                    solutions = [sent for sent in section["text"].split(".") if "solution" in sent.lower()]
                    for prob in problems:
                        prob_embedding = self.nlp.encode(prob, convert_to_tensor=True)
                        closest_solution = None
                        max_similarity = -1
                        for sol in solutions:
                            sol_embedding = self.nlp.encode(sol, convert_to_tensor=True)
                            similarity = util.pytorch_cos_sim(prob_embedding, sol_embedding)
                            if similarity > max_similarity:
                                max_similarity = similarity
                                closest_solution = sol
                        if closest_solution:
                            problem_solution_map.append({"problem": prob, "solution": closest_solution})

        return problem_solution_map

    def generate_and_save_dataset(self):
        """Generate and save a QA dataset."""
        dataset = []
        for section in self.sections:
            #if section["category"] == "qa":
            question = f"What is the solution for: {
                section['text'][:50]}..."
            answer = self.answer_question(question)
            dataset.append({
                "question": question,
                "answer": answer,
                "images": self.get_images_for_category(section["category"]),
                "summary": self.summarize_section(section["text"])
            })
        dataset_path = os.path.join(self.save_dir, "qa_dataset.json")
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=4)
        self.logger.info(f"Dataset saved to {dataset_path}.")

    def answer_question(self, question):
        """Answer a question using the QA pipeline."""
        context = " ".join(
            [sec["text"] for sec in self.sections if sec["category"] in ["qa", "repair"]])
        if not self.use_spacy and context:
            answer = self.qa_pipeline(question=question, context=context)
            return answer["answer"]
        return "Answer unavailable."

    def summarize_section(self, text):
        """Summarize text using a summarization model."""
        if not self.use_spacy:
            input_length = len(text.split())
            
            # Skip summarization for very short inputs
            if input_length < 50:
                return text
            
            # Dynamically adjust summarization length
            max_length = min(150, int(input_length * 0.8))
            min_length = max(50, int(input_length * 0.3))
            if min_length > max_length:
                min_length = max_length - 10 
            
            # Clean and truncate input text
            text = CoeusPdfProcessor.clean_text(text)
            max_tokens = 1024  # BART's token limit
            tokens = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
            truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            try:
                summary = self.summarizer(
                    truncated_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return summary[0]["summary_text"]
            except Exception as e:
                self.logger.error(f"Summarization failed: {e}")
                return text  # Return original text as fallback
        return text[:150] + "..."  # Fallback for SpaCy

    @staticmethod
    def clean_text(text):
        """Clean and normalize text."""
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text


    def get_images_for_category(self, category):
        """Get images associated with a category."""
        return [img["path"] for img in self.images if category in img["ocr_text"].lower()]

    def process_directory(self, directory_path):
        """Process multiple PDFs in a directory."""
        all_datasets = []
        threads = []

        # Batch processing with multi-threading
        def process_pdf(pdf_file):
            pdf_path = os.path.join(directory_path, pdf_file)
            print(f"Processing: {pdf_path}")
            self.logger.info(f"Processing: {pdf_path}")

            try:
                coeus_pdf = CoeusPdfProcessor(
                    pdf_path=pdf_path, save_dir=self.save_dir, to_dataset=True)
                with open(os.path.join(self.save_dir, "qa_dataset.json"), "r") as f:
                    all_datasets.extend(json.load(f))
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                self.logger.error(f"Error processing {pdf_path}: {e}")

        for pdf_file in os.listdir(directory_path):
            if pdf_file.endswith(".pdf"):
                thread = threading.Thread(target=process_pdf, args=(pdf_file,))
                threads.append(thread)
                thread.start()

        for thread in threads:
            thread.join()

        # Save all combined datasets
        with open(os.path.join(self.save_dir, "qa_dataset.json"), "w") as f:
            json.dump(all_datasets, f, indent=4)
        print(f"Combined dataset saved with {len(all_datasets)} question-answer pairs.")
        self.logger.info(f"Combined dataset saved with {len(all_datasets)} question-answer pairs.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PDF files to create QA datasets')
    parser.add_argument('--pdf_path', help='Path to a single PDF file')
    parser.add_argument('--save_dir', help='Directory to save the generated dataset')
    parser.add_argument('--directory_path', help='Directory containing multiple PDF files to process')
    parser.add_argument('--use_spacy', action='store_true', help='Use SpaCy for NLP tasks')

    args = parser.parse_args()

    if args.directory_path:
        processor = CoeusPdfProcessor(save_dir=args.save_dir)
        processor.process_directory(args.directory_path)
    elif args.pdf_path:
        processor = CoeusPdfProcessor(args.pdf_path, save_dir=args.save_dir, use_spacy=args.use_spacy)
        if not args.save_dir:
            processor.generate_and_save_dataset()
    else:
        parser.print_help()



#[2, 'GENERAL', 3, {'kind': 4, 'xref': 1665, 'page': '3', 'view': 'Fit', 'collapse': False, 'zoom': 0.0}]