from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import uuid
import random, re
import pandas as pd
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from classes import SubmittalPage, BOQItem, ItemVariant
from pydantic import ValidationError
from typing import List

# --- DOCS IMPORTS --
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.section import WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


def read_pdf(file_path):
    docs_reader = PyPDFLoader(file_path)
    docs = docs_reader.load()

    return docs

def extract_data_from_doc(file_path):
    pages = []
    docs = read_pdf(file_path)
    max_length = get_docs_max_length(docs)
    for page in docs:
      page_num = page.metadata['page'] + 1
      #Chunking
      page_chunks = text_chunks(page.page_content,max_length)


      #Creating Page Object
      contract = SubmittalPage(
          source = page.metadata['source'],
          page_num = page_num,
          content = page_chunks,
      )
      pages.append(contract)

    return pages

def get_docs_max_length(docs):
    #Check for max_length of single page
    max_length = 0
    for i in range(len(docs)):
        if len(docs[i].page_content) > max_length:
            max_length = len(docs[i].page_content)
            return max_length
        else:
            continue

def text_chunks(text,max_length):
  chunks = {}
  #Configure Langchain text splitter
  splitter_config = {
      "chunk_size": round((max_length/3),0),
      "chunk_overlap": max_length/10
  }
  text_splitter = RecursiveCharacterTextSplitter(**splitter_config)
  chunks=text_splitter.split_text(text)
  return chunks


def save_to_json(extracted_data):
    with open("processed_docs/processed_data.json","w",encoding= 'utf-8') as f:
        result = json.dumps(extracted_data,indent=4,ensure_ascii=False)
        f.write(result)

def save_pdf_to_collection(extracted_data,persistance_path,collection_name):
    client = PersistentClient(persistance_path)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name=collection_name,embedding_function=embedding_function)

    for id,page in enumerate(extracted_data):
        if page.content == []:
            continue
        else:
            pages = page.content
            ids = [str(uuid.uuid4()) for _ in pages]
            source = extracted_data[id].source.split("/")[-1]
            metadatas = [{"source":source, "page": page_num } for page_num in range(1, len(pages) + 1)]
            collection.upsert(
                documents=pages,
                ids=ids,
                metadatas=metadatas
            )

def save_boq_to_collection(extracted_data,persistance_path,collection_name):
    client = PersistentClient(persistance_path)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name=collection_name,embedding_function=embedding_function)

    metadatas = [{"item_num":item.itemNum, "variants_num": len(item.variants)} for item in extracted_data]
    ids = [str(uuid.uuid4()) for _ in extracted_data]
    documents = [item.model_dump_json() for item in extracted_data]

    collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

#Processing Excel files (BOQs)
def read_boq(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """Reads an Excel file and returns a pandas DataFrame."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        if df.empty:
            raise pd.errors.EmptyDataError(f"The file or sheet '{sheet_name}' is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"The file or sheet '{sheet_name}' is empty.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and transforms the DataFrame."""
    try:
        df = df.fillna("")
        if len(df.columns) < 6:
           raise ValueError("Input DataFrame does not have enough columns. Expected at least 6.")

        df.columns = ["Item", "Description", "Unit", "QTY", "Unit Price", "Total price"]
        #df = df.iloc[1:].reset_index(drop=True)
        df = df[df["Description"] != ""].reset_index(drop=True)
        df["Unit Price"] = [random.randint(500, 1000) for i in range(len(df["Unit Price"]))]
        df["Unit Price"] = pd.to_numeric(df["Unit Price"], errors='coerce').fillna(0.0)
        df["QTY"] = pd.to_numeric(df["QTY"], errors='coerce').fillna(0.0)
        df["Total price"] = df["QTY"] * df["Unit Price"]

        return df

    except KeyError as e:
        raise ValueError(f"Missing expected column(s) in DataFrame: {e}")
    except Exception as e:
        raise Exception(f"An error occurred during data processing: {e}")

def filter_chromadb_query_results(results,degree):
    filtered_results = []
    for i in range(len(results['distances'])):
        for j in range(len(results['distances'][i])):
            if results['distances'][i][j] < degree:
                filtered_results.append({
                    "id": results['ids'][i][j],
                    "document": results['documents'][i][j],
                    "metadata": results['metadatas'][i][j],
                    "distance": results['distances'][i][j]
                })
    return filtered_results


def retrieve_from_vectorstore(persistent_path, boq_collection_name, specs_collection_name,summarized_text):
  client = PersistentClient(persistent_path,settings=Settings(anonymized_telemetry=False))
  boq_collection = client.get_collection(name=boq_collection_name)
  specs_collection = client.get_collection(name=specs_collection_name)
  query_text = summarized_text

  results_specs = specs_collection.query(
      query_texts=[query_text],
      n_results=10,
  )

  results_boq = boq_collection.query(
      query_texts=[query_text],
      n_results=3,

  )

  filtered_results_boq = filter_chromadb_query_results(results_boq,1.3)
  filtered_results_specs = filter_chromadb_query_results(results_specs,1.3)
  query_boq_text = [filtered_results_boq[i]["document"] for i in range(len(filtered_results_boq))]
  query_specs_text = [filtered_results_specs[i]["document"] for i in range(len(filtered_results_specs))]
  return query_boq_text, query_specs_text

def datasheet_content(data_sheet_file_path, model):
  submittal_data_text = PyPDFLoader(data_sheet_file_path).load()
  submittal_data_text = [submittal_data_text[i].page_content for i in range(len(submittal_data_text))]
  submittal_data_text = "\n".join(submittal_data_text)

  data_sheet_summarize_query = """
  You are a helpful assistant, I want you to summarize the following text and return only the summarized text without any additional texts in the beginning or at the end.

  Text : {text}

  Note that you should extract the following information from the text:
  1-specifications
  2-design characteristics
  3-general features
  4-data sheets
  5-any important or valuable info

  **Do not miss any information ( the summarization purpose is to organize the information in the datasheet for further processing
  """.format(text = "\n".join(submittal_data_text))

  summarized_text = model.invoke(data_sheet_summarize_query).content
  return submittal_data_text

def group_boq_items(df: pd.DataFrame, delimiter: str = "-") -> pd.core.groupby.DataFrameGroupBy:
    """Groups DataFrame rows by item number, using regex for more robust matching."""
    try:
        df["item_group_num"] = ""
        # Regex to match the first part of the item number (digits before the delimiter)
        item_number_pattern = re.compile(r"^(\d+)")

        for index, row in df.iterrows():
            item_str = str(row["Item"])
            if item_str != "":
                match = item_number_pattern.match(item_str)
                if match:
                    number_str = match.group(1)  # Get the captured group (the digits)
                    try:
                        df.loc[index, "item_group_num"] = int(number_str)
                    except ValueError:
                        df.loc[index, "item_group_num"] = -1
                        print(f"Warning: Invalid item number format at row {index}: {number_str}")
                else:
                    df.loc[index, "item_group_num"] = -1
                    print(f"Warning: Invalid item number format at row {index}: {item_str}")

        df_group = df.groupby("item_group_num")
        return df_group

    except KeyError as e:
        raise ValueError(f"Missing expected column(s) during grouping: {e}")
    except Exception as e:
        raise Exception(f"An error occurred during item grouping: {e}")


def extract_boq_data(df_group: pd.core.groupby.DataFrameGroupBy) -> List[BOQItem]:
    """Extracts BOQ data from grouped DataFrame."""
    boq_item_list: List[BOQItem] = []
    try:
        for item_group_num, item_group_df in df_group:
            variant_list: List[ItemVariant] = []
            if len(item_group_df) > 1:
                for i in range(1, len(item_group_df)):
                    row = item_group_df.iloc[i]
                    try:
                        item_variant = ItemVariant(
                            variant_num=str(row["Item"]),
                            description=str(row["Description"]),
                            unit=str(row["Unit"]),
                            quantity=row["QTY"],
                            rate=row["Unit Price"],
                            amount=row["Total price"]
                        )
                        variant_list.append(item_variant)
                    except ValidationError as ve:
                        print(
                            f"Validation error for ItemVariant at row {i} in group {item_group_num}: {ve}")
                        continue
                    except Exception as e:
                        print(f"Error creating ItemVariant at row {i} in group {item_group_num}: {e}")
                        continue

                main_row = item_group_df.iloc[0]
                try:
                    boq_item = BOQItem(
                        itemNum=str(main_row["Item"]),
                        description=list(item_group_df["Description"].astype(str)),
                        unit=str(main_row["Unit"]),
                        quantity=main_row["QTY"],
                        rate=main_row["Unit Price"],
                        amount=main_row["Total price"],
                        variants=variant_list
                    )
                    boq_item_list.append(boq_item)
                except ValidationError as ve:
                    print(f"Validation error for BOQItem in group {item_group_num}: {ve}")
                except Exception as e:
                    print(f"Error creating BOQItem in group {item_group_num}: {e}")


            else:
                main_row = item_group_df.iloc[0]
                try:
                    boq_item = BOQItem(
                        itemNum=str(main_row["Item"]),
                        description=list(item_group_df["Description"].astype(str)),
                        unit=str(main_row["Unit"]),
                        quantity=main_row["QTY"],
                        rate=main_row["Unit Price"],
                        amount=main_row["Total price"],
                        variants=[]
                    )
                    boq_item_list.append(boq_item)
                except ValidationError as ve:
                    print(f"Validation error for BOQItem in single-line group {item_group_num}: {ve}")
                except Exception as e:
                    print(f"Error creating BOQItem in single-line group {item_group_num}: {e}")

    except Exception as e:
        raise Exception(f"An unexpected error occurred during data extraction: {e}")

    return boq_item_list

def generate_report(boq_retrieved_docs,specs_retrieved_docs, summarized_text, model):
  with open("prompt_gallery\generate_report.txt","r") as f:
    query = f.read()
  query = query.format(specs = specs_retrieved_docs, boq = boq_retrieved_docs, submittal_text = summarized_text)
  model_response = model.invoke(query)
  return model_response

def create_docx_from_markdown(markdown_text, output_filename):
    """
    Creates a .docx file from the provided markdown text with landscape layout,
    modern styles, margins, adjusted line spacing, and a page border.

    Args:
        markdown_text: The markdown text to convert.
        output_filename: The name of the output .docx file.
    """

    document = Document()

    # --- Document Setup (Landscape, Margins, Styles) ---
    section = document.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11.69)  # A4 Landscape width
    section.page_height = Inches(8.27)  # A4 Landscape height
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)

    # Create or access a style for normal text
    normal_style = document.styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Calibri'  # Modern font
    normal_font.size = Pt(11)
    normal_paragraph_format = normal_style.paragraph_format
    normal_paragraph_format.space_before = Pt(0)  # Remove space before paragraph
    normal_paragraph_format.space_after = Pt(0)   # Remove space after paragraph
    normal_paragraph_format.line_spacing = 1.15      # Set line spacing to 1.15 (adjust as needed)



    # --- Helper Functions ---
    def add_paragraph(text, style='Normal', alignment=None):
        p = document.add_paragraph(text, style=style)
        if alignment:
            p.alignment = alignment
        return p

    def add_heading(text, level):
        heading = document.add_heading(text, level=level)
        heading.style.font.name = 'Calibri'
        heading.paragraph_format.space_before = Pt(0)  # Remove space before heading
        heading.paragraph_format.space_after = Pt(6)   # Add a little space after heading

        return heading

    # --- Table Style ---
    table_style = document.styles.add_style('CustomTableStyle', WD_STYLE_TYPE.TABLE)
    table_style.base_style = document.styles['Table Grid']
    table_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # --- Add Page Border ---
    def create_element(name):
        return OxmlElement(name)

    def create_attribute(element, name, value):
        element.set(qn(name), value)

    page_border_elm = create_element('w:pPr')
    page_border_props = create_element('w:pBdr')
    page_border_top = create_element('w:top')
    create_attribute(page_border_top, 'w:val', 'single')
    create_attribute(page_border_top, 'w:sz', '4')
    create_attribute(page_border_top, 'w:space', '24')
    create_attribute(page_border_top, 'w:color', 'auto')
    page_border_props.append(page_border_top)

    page_border_left = create_element('w:left')
    create_attribute(page_border_left, 'w:val', 'single')
    create_attribute(page_border_left, 'w:sz', '4')
    create_attribute(page_border_left, 'w:space', '24')
    create_attribute(page_border_left, 'w:color', 'auto')
    page_border_props.append(page_border_left)

    page_border_bottom = create_element('w:bottom')
    create_attribute(page_border_bottom, 'w:val', 'single')
    create_attribute(page_border_bottom, 'w:sz', '4')
    create_attribute(page_border_bottom, 'w:space', '24')
    create_attribute(page_border_bottom, 'w:color', 'auto')
    page_border_props.append(page_border_bottom)

    page_border_right = create_element('w:right')
    create_attribute(page_border_right, 'w:val', 'single')
    create_attribute(page_border_right, 'w:sz', '4')
    create_attribute(page_border_right, 'w:space', '24')
    create_attribute(page_border_right, 'w:color', 'auto')
    page_border_props.append(page_border_right)

    page_border_elm.append(page_border_props)

    # Apply border to each paragraph in the document.  This will appear as a page border.
    for paragraph in document.paragraphs:
      paragraph._element.insert_element_before(page_border_elm, 'w:pPr')


    # --- Markdown Parsing ---
    lines = markdown_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("## "):
            add_heading(line[3:], level=2)
        elif line.startswith("**"):
            add_paragraph(line[2:].strip(), style='Normal')
        elif line.startswith("|---"):
            # Table detected, parse the table
            header_line = lines[i - 1].strip()
            data_lines = []
            i += 1  # Skip the separator line
            while i < len(lines) and lines[i].strip().startswith("|"):
                data_lines.append(lines[i].strip())
                i += 1
            i -= 1 # Decrement i because the loop ends when it reaches a non-table line

            header_cells = [cell.strip() for cell in header_line.split("|")[1:-1]]  # Remove leading/trailing pipes
            data_rows = []
            for data_line in data_lines:
                data_cells = [cell.strip() for cell in data_line.split("|")[1:-1]]  # Remove leading/trailing pipes
                data_rows.append(data_cells)


            table = document.add_table(rows=1, cols=len(header_cells))
            table.style = 'CustomTableStyle'

            # Add header row
            header_cells = [h.replace('Requirement/Feature', 'Requirement') for h in header_cells] # replace Requirement/Feature with Requirement to avoid error in heading
            for j, header_text in enumerate(header_cells):
                cell = table.cell(0, j)
                cell.text = header_text
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER  # Center align headers
                cell.paragraphs[0].style.font.bold = True  # Bold header text

            # Add data rows
            for row_index, row_data in enumerate(data_rows):
                row_cells = table.add_row().cells
                for j, cell_data in enumerate(row_data):
                    row_cells[j].text = cell_data
                    row_cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER  # Center align content

            # Adjust column widths for better readability (optional)
            for col in table.columns:
                for cell in col.cells:
                    cell.width = Inches(1.5)  # Adjust width as needed
                    for paragraph in cell.paragraphs:
                        paragraph.style.font.name = 'Cambria' # Apply Calibri font to table content
                        paragraph.paragraph_format.space_before = Pt(0) #Remove space before paragraph
                        paragraph.paragraph_format.space_after = Pt(0) #Remove space after paragraph
                        paragraph.paragraph_format.line_spacing = 1.15 #set line spacing to 1.15


        else:
            add_paragraph(line)

        i += 1

    # --- Save the document ---
    document.save(output_filename)
    return 