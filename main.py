import io, re, random, string, os
import zipfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract

app = FastAPI(title="Invoice → Bill of Lading Generator")

# -------------------------------------------------------
# CORS
# -------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# PDF TEXT EXTRACTION
# -------------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF using pdfplumber; fallback to OCR if scanned."""
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    if not text.strip():  # OCR fallback
        for page_img in convert_from_bytes(pdf_bytes):
            text += pytesseract.image_to_string(page_img) + "\n"
    return text.strip()

# -------------------------------------------------------
# DATA EXTRACTION (Regex)
# -------------------------------------------------------
def extract_invoice_data(text: str) -> dict:
    """Extract structured invoice data using robust regex; fallback randoms for missing fields."""
    raw = text or ""
    FLAGS = re.IGNORECASE | re.DOTALL

    # Use the raw text for matching
    def find(pattern: str, source: str = None):
        s = raw if source is None else source
        m = re.search(pattern, s, FLAGS)
        try:
            return (m.group(1) or "").strip() if m else ""
        except Exception:
            return ""

    def extract_block(label_regex: str, next_labels_regex: str, source: str = None) -> str:
        s = raw if source is None else source
        pattern = rf'{label_regex}\s*[:\-]?\s*(.*?)(?={next_labels_regex}|$)'
        m = re.search(pattern, s, FLAGS)
        block = (m.group(1) or "").strip() if m else ""
        # restore line breaks where multiple spaces may exist
        block = re.sub(r'\n\s*', '\n', block)
        return block

    # Build a union of label starters for lookahead boundaries
    label_alts = [
        r'Invoice\s*No\.?', r'I\.?E\.?\s*Code', r"Buyer'?s\s*Order\s*No\.",
        r'Consignee', r'Notify\s*Party', r'Country', r'Pre-?Carriage', r'Vessel\s*\/?\s*Voyage',
        r'Place\s*of\s*Receipt', r'Port\s*of\s*Loading', r'Port\s*of\s*Discharge', r'Final\s*Destination',
        r'Terms\s*of\s*Payment', r'Amount\s*Chargeable', r'Total', r'BIN\s*NO', r'Drawback', r'Benefit', r'Shipment'
    ]
    boundary = r'(?=' + r'|'.join(label_alts) + r')'

    # 🧾 BASIC DETAILS
    invoice_no = find(r'Invoice\s*No\.?\s*[:\-]?\s*([A-Z0-9\-\/]+)')
    ie_code = find(r'I\.?E\.?\s*Code\s*No\.?\s*[:\-]?\s*([A-Z0-9]+)')
    po_no = find(r'Buyer.?s\s*Order\s*No\.?\s*[:\-]?\s*([A-Z0-9\-\/]+)')
    terms = find(r'Terms\s*of\s*Payment\s*[:\-]?\s*(.+?)(?:Country|Final|Port|$)')
    drawback_no = find(r'Drawback\s*Sr\.?\s*No\.?\s*[:\-]?\s*([A-Z0-9\-\/]+)')
    benefit_scheme = find(r'Benefit[s]?\s*under\s*ME[I|E]S\s*scheme\s*[:\-]?\s*([A-Za-z ]+)')
    total = find(r'(?:Amount\s*Chargeable|Total)\s*[:\-]?\s*(?:USD\s*)?([\d,]+(?:\.\d{2})?)')
    currency = "USD" if re.search(r'USD', raw, re.IGNORECASE) else "NOT FOUND"

    # 🏢 EXPORTER / CONSIGNEE / NOTIFY PARTY (multiline blocks)
    # Shipper is same as Exporter
    exporter_block = extract_block(r'Exporter|Shipper', boundary)
    consignee_block = extract_block(r'Consignee', boundary)
    notify_block = extract_block(r'Notify\s*Party', boundary)

    # Fallbacks: if blocks are empty, try broader spans between common labels
    if not exporter_block:
        m = re.search(r'(?:Exporter|Shipper)\s*:\s*([\s\S]{10,800}?)(?=Consignee|Notify\s*Party|Invoice\s*No\.?|Country|Pre-?Carriage|$)', raw, FLAGS)
        exporter_block = (m.group(1).strip() if m else "")
    if not consignee_block:
        m = re.search(r'Consignee\s*:\s*([\s\S]{10,800}?)(?=Notify\s*Party|Country|Pre-?Carriage|Invoice\s*No\.?|$)', raw, FLAGS)
        consignee_block = (m.group(1).strip() if m else "")
        # Additional fallback for consignee
        if not consignee_block:
            m = re.search(r'Consignee\s*[:\-]?\s*([\s\S]{10,500}?)(?=Notify|Country|Port|Vessel|$)', raw, FLAGS)
            consignee_block = (m.group(1).strip() if m else "")


    # Split exporter into name/address. Prefer first meaningful company-like line.
    exporter_name = ""
    exporter_address = ""
    if exporter_block:
        lines = [ln.strip() for ln in re.split(r'[\r\n]+', exporter_block) if ln.strip()]
        if lines:
            # pick first candidate that looks like a company (not 'INVOICE')
            candidate = None
            for ln in lines:
                if re.search(r'Invoice|Bill\s*of\s*Lading|B/L|Date|Tax|GST', ln, re.IGNORECASE):
                    continue
                candidate = ln
                break
            if not candidate:
                candidate = lines[0]
            exporter_name = re.sub(r'(?mi)^(?:Exporter|Shipper)\s*:?\s*', '', candidate).strip()
            exporter_address = "\n".join([l for l in lines if l != candidate]).strip()

    # If exporter name is missing or is a generic placeholder, try to extract from top of document
    if not exporter_name or exporter_name.strip().lower() in ('invoice', 'exporter', 'shipper', 'seller'):
        m = re.search(r'^(.*?)(?=\n\s*(?:Consignee|Notify\s*Party|Invoice\s*No\.?|I\.?E\.?\s*Code|$))', raw, re.IGNORECASE | re.DOTALL)
        if m:
            top_block = m.group(1).strip()
            top_lines = [ln.strip() for ln in re.split(r'[\r\n]+', top_block) if ln.strip()]
            for ln in top_lines:
                if len(ln) > 2 and not re.search(r'Invoice|Bill\s*of\s*Lading|B/L|Date', ln, re.IGNORECASE):
                    exporter_name = exporter_name or ln
                    # set remaining as address
                    rest = [l for l in top_lines if l != ln]
                    exporter_address = exporter_address or "\n".join(rest).strip()
                    break
        exporter_block = re.sub(r'(?mi)^.*\bI\.?E\.?\s*Code.*$\n?', '', exporter_block)
        # Remove Invoice No / Buyer's Order No lines if accidentally captured
        exporter_block = re.sub(r'(?mi)^.*\bInvoice\s*No\.?\b.*$\n?', '', exporter_block)
        exporter_block = re.sub(r"(?mi)^.*Buyer'?s\s*Order\s*No\.?\b.*$\n?", '', exporter_block)
        # Remove leading 'Exporter:'/'Shipper:' labels if present
        exporter_block = re.sub(r'(?mi)^(?:Exporter|Shipper)\s*:?\s*', '', exporter_block)
        # Collapse excessive blank lines
        exporter_block = re.sub(r'\n{3,}', '\n\n', exporter_block)
        exporter_block = exporter_block.strip()

        # Remove plain PO lines like 'PO-123456' which sometimes appear in exporter block
        if exporter_block:
            exporter_block = re.sub(r'(?mi)^\s*PO[-\s]*\d+\b.*$\n?', '', exporter_block)
            exporter_block = exporter_block.strip()

    # --- Split exporter block into name + address for structured output ---
    exporter_name = ""
    exporter_address = ""
    if exporter_block:
        lines = [ln.strip() for ln in re.split(r'[\r\n]+', exporter_block) if ln.strip()]
        if lines:
            # Prefer a company-like line (contains company tokens or uppercase short name)
            company_token = re.compile(r'\b(IMPEX|PVT|LTD|LLP|LLC|TRADING|EXPORTS|IMPORTS|CO\.|COMPANY|INDIA)\b', re.IGNORECASE)
            candidate = None
            for ln in lines:
                if company_token.search(ln) or re.match(r'^[A-Z0-9 &,\-\.]{{3,}}$', ln):
                    candidate = ln
                    break
            # if none found, prefer a short non-address first line
            if not candidate:
                for ln in lines:
                    if not looks_like_address(ln) and len(ln) <= 60:
                        candidate = ln
                        break
            # fallback to first line
            if not candidate:
                candidate = lines[0]
            # remaining lines are address
            remaining = [l for l in lines if l != candidate]
            exporter_name = candidate
            exporter_address = "\n".join(remaining).strip()
        # Fallbacks
        exporter_name = exporter_name or (lines[0] if lines else exporter_block)
        exporter_address = exporter_address or ("\n".join(lines[1:]) if len(lines) > 1 else "")

    # If exporter_name is missing or appears to be an address (e.g. starts with 'Plot', contains 'GIDC' or long numeric address),
    # try a fallback: take the top-of-document block until common labels (Consignee/Notify/Invoice) and use its first line as name.
    def looks_like_address(s: str) -> bool:
        return bool(re.search(r'\bPlot\b|\bGIDC\b|\bIndustrial\b|\bEstate\b|\bPhase\b|\bIndia\b|\d{3,}', s, re.IGNORECASE))

    if not exporter_name or looks_like_address(exporter_name):
        top_block = ""
        m_top = re.search(r'^(.*?)(?=\bConsignee\b|\bNotify\s*Party\b|\bInvoice\s*No\.?|\bBuyer\'?s\s*Order\b|\bCountry\b)', raw, FLAGS)
        if m_top:
            top_block = m_top.group(1).strip()
        # Clean common headings from the top block
        if top_block:
            top_lines = [ln.strip() for ln in re.split(r'[\r\n]+', top_block) if ln.strip()]
            # remove leading labels/headings if present
            if top_lines and re.search(r'^(Exporter|Shipper|INVOICE|BILL|TAX)', top_lines[0], re.IGNORECASE):
                # drop obvious heading line
                top_lines = top_lines[1:]

            # helper to detect likely company name lines
            def is_company_line(s: str) -> bool:
                if not s or len(s) < 3:
                    return False
                # exclude very generic single words
                deny = [r'INVOICE', r'BILL', r'TAX', r'GST', r'AMOUNT', r'DATE']
                if any(re.search(d, s, re.IGNORECASE) for d in deny):
                    return False
                # must contain letters and at least one space (two words) and not look like an address
                if re.search(r'[A-Za-z]', s) and len(s.split()) >= 1 and not looks_like_address(s):
                    # prefer shortish company-like lines
                    return len(s) <= 80
                return False

            candidate = None
            for ln in top_lines:
                if is_company_line(ln):
                    candidate = ln
                    break
            if not candidate and top_lines:
                # fallback to first non-empty top line
                candidate = top_lines[0]

            if candidate:
                exporter_name = candidate
                # exporter_address: remaining top lines excluding the chosen candidate and any heading-like lines
                remaining = [ln for ln in top_lines if ln != candidate and not re.search(r'^(Exporter|Shipper|INVOICE|BILL|TAX)', ln, re.IGNORECASE)]
                exporter_address = exporter_address or "\n".join(remaining).strip()

    # Extra fallback: sometimes OCR/text layouts put a standalone 'INVOICE' or other heading first.
    # If exporter_name is still missing or equals 'INVOICE', scan the raw text before 'Consignee' for a company-like line.
    if not exporter_name or re.search(r'INVOICE', exporter_name, re.IGNORECASE):
        pre_cons_block = raw
        m_cons = re.search(r'(.*?)(?=\bConsignee\b|\bNotify\s*Party\b|\bInvoice\s*No\.?|\bBuyer\'??s\s*Order\b)', raw, FLAGS)
        if m_cons:
            pre_cons_block = m_cons.group(1)
        candidate = None
        for ln in [l.strip() for l in re.split(r'[\r\n]+', pre_cons_block) if l.strip()]:
            # skip generic headings
            if re.search(r'^(INVOICE|BILL|TAX|STATEMENT)$', ln, re.IGNORECASE):
                continue
            # prefer lines containing common company tokens or uppercase style
            if re.search(r'\b(IMPEX|PVT|LTD|LLC|COMPANY|TRADING|EXPORTS|IMPORTS|INDIA|GMBH)\b', ln, re.IGNORECASE) or re.match(r'^[A-Z0-9 &,\-\.]{3,}$', ln):
                candidate = ln
                break
            # Title-case company name (like 'Shraddha Impex')
            if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)+', ln):
                candidate = ln
                break
        if candidate:
            exporter_name = candidate
            # build address from lines after candidate
            all_lines = [l.strip() for l in re.split(r'[\r\n]+', pre_cons_block) if l.strip()]
            try:
                idx = all_lines.index(candidate)
                exporter_address = exporter_address or "\n".join(all_lines[idx+1:]).strip()
            except ValueError:
                exporter_address = exporter_address or ""

    # FINAL FALLBACK: search anywhere in raw for a company-like token (IMPEX/PVT/LTD/LLC/TRADING)
    # This helps when earlier block parsing misses a top-line company name.
    if not exporter_name:
        company_match = re.search(r'^(.*(?:IMPEX|PVT|LTD|LLP|LLC|TRADING|EXPORTS|IMPORTS|COMPANY|CO\.)[^\n]*)$', raw, re.IGNORECASE | re.MULTILINE)
        if company_match:
            candidate = company_match.group(1).strip()
            # avoid accidental headings
            if not re.search(r'^(INVOICE|BILL|TAX)$', candidate, re.IGNORECASE):
                exporter_name = candidate
                # try to set exporter_address from nearby lines (lines following the candidate)
                all_lines = [l.strip() for l in re.split(r'[\r\n]+', raw) if l.strip()]
                try:
                    idx = all_lines.index(candidate)
                    exporter_address = exporter_address or "\n".join(all_lines[idx+1:idx+5]).strip()
                except ValueError:
                    pass

    # EXTRA: if still not found, try a simple labeled 'Exporter:' line (common layouts)
    if not exporter_name:
        m = re.search(r'Exporter\s*[:\-]?\s*(.+)', raw, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().split('\n')[0].strip()
            if candidate and not re.search(r'^(INVOICE|BILL|TAX)$', candidate, re.IGNORECASE):
                exporter_name = candidate
                # try to set exporter_address from the exporter_block if present
                if exporter_block and not exporter_address:
                    lines = [ln.strip() for ln in re.split(r'[\r\n]+', exporter_block) if ln.strip()]
                    if lines and lines[0] == candidate:
                        exporter_address = "\n".join(lines[1:]).strip()

    # Notify Party: mirror Consignee if missing or marked same as consignee
    if (not notify_block or re.search(r'same as consignee', notify_block, re.IGNORECASE)):
        notify_block = consignee_block

    # Use notify party data for consignee if consignee is empty
    if not consignee_block or consignee_block.strip() == "":
        consignee_block = notify_block

    # Remove embedded 'Notify Party' labels inside consignee block (we already have a CONSIGNEE header)
    if consignee_block:
        consignee_block = re.sub(r'(?mi)^Notify\s*Party\s*:?\s*', '', consignee_block)
        consignee_block = re.sub(r'(?mi)^Notify\s*Party\s*[:\-]?\s*', '', consignee_block)

    # 🚢 SHIPMENT DETAILS
    pre_carriage = find(r'(?:Pre|Pre\-)?\s*Carriage\s*By\s*[:\-]?\s*([A-Za-z \-]+)') or find(r'Pre\-?Carriage\s*[:\-]?\s*([A-Za-z \-]+)')
    vessel_voyage = find(r'Vessel\s*\/?\s*Voyage\s*[:\-]?\s*([A-Za-z0-9 .\-\/]+)')
    # Use block extraction for all four header fields with robust lookahead boundaries
    por_block = extract_block(r'Place\s*of\s*receipt|Place\s*of\s*Acceptance', boundary)
    pl_block = extract_block(r'Port\s*of\s*Loading|Port\s*of\s*Shipment', boundary)
    pd_block = extract_block(r'Port\s*of\s*Discharge', boundary)
    podl_block = extract_block(r'Place\s*of\s*Delivery|Final\s*Destination', boundary)
    # Specific regex patterns for known ports (use capturing groups so finder returns value)
    pol_direct = (
        find(r'(Nhava\s*Sheva|Nava\s*Sheva|Nahava\s*Seva|JNPT)') or
        find(r'Port\s*of\s*Loading\s*[:\-]?\s*([^\n]+)') or 
        find(r'Port\s*of\s*Shipment\s*[:\-]?\s*([^\n]+)') or
        find(r'Loading\s*Port\s*[:\-]?\s*([^\n]+)') or
        find(r'POL\s*[:\-]?\s*([^\n]+)')
    )
    pod_direct = (
        find(r'(Singapore)') or
        find(r'Port\s*of\s*Discharge\s*[:\-]?\s*([^\n]+)') or 
        find(r'Port\s*of\s*Delivery\s*[:\-]?\s*([^\n]+)') or
        find(r'Discharge\s*Port\s*[:\-]?\s*([^\n]+)') or
        find(r'POD\s*[:\-]?\s*([^\n]+)')
    )
    def first_line(val: str) -> str:
        if not val:
            return ''
        for line in val.split('\n'):
            s = line.strip(' :\t')
            if s:
                return s
        return ''
    place_receipt = (first_line(por_block) or find(r'(?:Place|Place\s*of)\s*of?\s*Receipt\s*[:\-]?\s*([^\n]+)') or find(r'Place\s*of\s*Acceptance\s*[:\-]?\s*([^\n]+)'))
    port_loading = (pol_direct or first_line(pl_block))
    port_discharge = (pod_direct or first_line(pd_block))
    final_destination = (first_line(podl_block) or find(r'Final\s*Destination\s*[:\-]?\s*([^\n]+)') or find(r'Place\s*of\s*Delivery\s*[:\-]?\s*([^\n]+)'))
    country_origin = find(r'Country\s*of\s*Origin\s*[:\-]?\s*([A-Za-z ,]+)')
    country_destination = find(r'Country\s*of\s*Final\s*Destination\s*[:\-]?\s*([A-Za-z ,]+)')
    # Container & Seal (fallback from raw text)
    cont_seal_match = re.search(r'Container\s*&\s*Seal\s*nos?\.?\s*[:\-]?\s*([A-Z0-9\/\-]+)\s*[\/\-\| ]\s*([A-Z0-9]+)', raw, FLAGS)
    if cont_seal_match:
        container_no = cont_seal_match.group(1).strip()
        seal_no = cont_seal_match.group(2).strip()

    # 📦 GOODS EXTRACTION
    goods_matches = re.findall(
        r'HS\s*CODE\s*:\s*([0-9\.]+)[\s\S]*?QUANTITY\s*:\s*([0-9,]+\s*PCS)[\s\S]*?WEIGHT\s*:\s*([0-9,\.]+\s*KGS?)[\s\S]*?PACKING\s*:\s*([0-9,]+\s*CARTONS?)',
        text, FLAGS
    )

    goods = []
    # Attempt to capture Sr No & Marks / Containers block (optional)
    sr_marks_block_match = re.search(r'(\d+\s*X\s*20[\'\’\″\”\"]?\s*FCL[\s\S]*?Container\s*&\s*Seal\s*nos\.?\s*:\s*[\s\S]*?)(?:INDIAN|HS\s*CODE|Description|NO\.|$)', raw, FLAGS)
    sr_marks_block = sr_marks_block_match.group(1).strip() if sr_marks_block_match else ""
    # If not found, try a simpler pattern like "06 X 20' FCL ..."
    if not sr_marks_block:
        simple_fcl = re.search(r'(\d+\s*X\s*20[\'\’\″\”\"]?\s*FCL[^\n]*?)', raw, FLAGS)
        if simple_fcl:
            sr_marks_block = simple_fcl.group(1).strip()
    # If still not found but container/seal extracted, compose sr_marks text
    if not sr_marks_block and (locals().get('container_no') and locals().get('seal_no')):
        sr_marks_block = f"Container & Seal nos.: {container_no} / {seal_no}"
    # Clean sr_marks to avoid product/description lines leaking into left column
    if sr_marks_block:
        sr_lines = [ln.strip() for ln in re.split(r'[\r\n]+', sr_marks_block) if ln.strip()]
        allowed_patterns = [r'\bFCL\b', r'Container', r'Seal', r'Marks', r'Packages?', r'Pkg', r'Cartons?', r'\bNo\.?']
        def is_allowed(line: str) -> bool:
            return any(re.search(p, line, re.IGNORECASE) for p in allowed_patterns)
        sr_filtered = []
        # Exclude product/description-like lines (ICUMSA, PACKED, NET WEIGHT, HS CODE, measurements etc.)
        product_exclude_patterns = [r'\bICUMSA\b', r'PACK', r'NET\s*WEIGHT', r'TOTAL', r'GROSS', r'HS\s*CODE', r'\bKGS?\b', r'\bMTS?\b', r'BAGS?', r'PP\s*BAGS', r'PACKED', r'WEIGHT']
        def is_product_line(line: str) -> bool:
            return any(re.search(p, line, re.IGNORECASE) for p in product_exclude_patterns)

        for ln in sr_lines:
            # skip lines that clearly look like product descriptions or measurements
            if is_product_line(ln):
                continue
            # capture FCL counts like '06 X 20' FCL' as compact token
            if re.search(r'\d+\s*X\s*20[\'\’\″\”\"]?\s*FCL', ln, re.IGNORECASE):
                m = re.search(r'(\d+\s*X\s*20[\'\’\″\”\"]?\s*FCL)', ln, re.IGNORECASE)
                if m:
                    sr_filtered.append(m.group(1))
                continue
            if is_allowed(ln):
                sr_filtered.append(ln)
        # If filtering removed everything but container/seal exists, ensure container & seal line remains
        if not sr_filtered and (locals().get('container_no') and locals().get('seal_no')):
            sr_filtered = [f"Container & Seal nos.: {container_no} / {seal_no}"]
        sr_marks_block = "\n".join(sr_filtered).strip()

    # Capture Units (In Metric Tons), Rate Per Unit (USD), Amount (USD) from headers if present
    units_mt = find(r'NO\.\s*OF\s*UNITS\s*\(In\s*Metric\s*Tons\)\s*([0-9,.]+)')
    if units_mt == "NOT FOUND":
        # fallback to NET WEIGHT value in MTS
        units_mt = find(r'TOTAL\s*NET\s*WEIGHT\s*[:\-]?\s*([0-9,.]+)\s*MTS?')

    rate_per_unit = find(r'RATE\s*PER\s*UNIT\s*\(USD\)\s*([0-9,.]+)')
    amount_usd = find(r'Amount\s*\(USD\)\s*([0-9,.]+)')
    total_net_wt = find(r'TOTAL\s*NET\s*WEIGHT\s*[:\-]?\s*([0-9,.]+)\s*(?:MTS?|KGS?)')
    total_gross_wt = find(r'TOTAL\s*GROSS\s*WEIGHT\s*[:\-]?\s*([0-9,.]+)\s*(?:MTS?|KGS?)')
    measurement_cbm = find(r'MEASUREMENT\s*[:\-]?\s*([0-9,.]+\s*CBM)')
    if amount_usd == "NOT FOUND":
        amount_usd = find(r'Total\s*[:\-]?\s*([0-9,.]+)')
    for match in goods_matches:
        hs, qty, weight, pack = match
        desc_match = re.search(rf'HS\s*CODE\s*:\s*{re.escape(hs)}\s*([\s\S]*?)\s*QUANTITY', raw, FLAGS)
        try:
            desc_raw = desc_match.group(1) if desc_match else ""
        except Exception:
            desc_raw = ""
        # Try to capture 1-2 lines just before the HS CODE (product names)
        pre_hs_name = ""
        # Try to capture 1-2 lines before HS CODE; handle cases with or without newline
        pre_hs_context = re.search(rf'([^\r\n]{{8,240}})\s*[\r\n]+\s*HS\s*CODE\s*[:\-]?\s*{re.escape(hs)}', raw, FLAGS)
        if not pre_hs_context:
            pre_hs_context = re.search(rf'([^\r\n]{{8,240}})\s*HS\s*CODE\s*[:\-]?\s*{re.escape(hs)}', raw, FLAGS)
        if pre_hs_context:
            pre_hs_name = pre_hs_context.group(1).strip()
        # Preserve content and newlines (trim excessive length)
        combined_desc = (pre_hs_name + "\n" + (desc_raw or "")).strip()
        if combined_desc:
            combined_desc = combined_desc[:2200]
            desc = re.sub(r'[ \t]{2,}', ' ', combined_desc).strip()
        else:
            desc = ""
        desc_full = desc
        # Append totals to description body as lines
        if total_net_wt or total_gross_wt:
            tail = []
            if total_net_wt:
                tail.append(f"TOTAL NET WEIGHT: {total_net_wt} MTS")
            if total_gross_wt:
                tail.append(f"TOTAL GROSS WEIGHT: {total_gross_wt} MTS")
            desc_full = (desc + "\n" + "\n".join(tail)).strip()
        # Prepend HS CODE to description per request
        if hs:
            desc_full = (f"HS CODE: {hs}\n" + desc_full).strip()
        # Compose weight & measurements text for right column
        wm_lines = []
        if total_net_wt:
            wm_lines.append(f"NET: {total_net_wt} MTS")
        if total_gross_wt:
            wm_lines.append(f"GROSS: {total_gross_wt} MTS")
        if measurement_cbm and measurement_cbm != "NOT FOUND":
            wm_lines.append(f"MEASUREMENT: {measurement_cbm}")
        # If totals missing, fallback to WEIGHT field captured from goods section
        if not wm_lines and weight:
            wm_lines.append(f"WEIGHT: {weight}")
        weight_measurements = "\n".join(wm_lines)
        # Append structured fields to description for completeness
        extra = []
        if qty:
            extra.append(f"QUANTITY: {qty}")
        if weight:
            extra.append(f"WEIGHT: {weight}")
        if pack:
            extra.append(f"PACKING: {pack}")
        if amount_usd and amount_usd != "NOT FOUND":
            extra.append(f"AMOUNT(USD): {amount_usd}")
        if extra:
            desc_full = (desc_full + "\n" + "\n".join(extra)).strip()
        goods.append({
            "hs_code": hs,
            "description": desc_full,
            "quantity": qty,
            "weight": weight,
            "packing": pack,
            "unit": "PCS",
            "rate": rate_per_unit if rate_per_unit != "NOT FOUND" else "",
            "amount": amount_usd if amount_usd != "NOT FOUND" else "",
            "units_mt": units_mt if units_mt != "NOT FOUND" else "",
            "weight_measurements": weight_measurements,
            "sr_marks": sr_marks_block
        })

    # Fallback: common invoice layout (as in your screenshot)
    # Capture description block around HS CODE and TOTAL NET WEIGHT lines
    if not goods:
        hs_only = re.search(r'HS\s*CODE\s*[:\-]?\s*([0-9\.]{6,10})', raw, FLAGS)
        total_net_match = re.search(r'TOTAL\s*NET\s*WEIGHT\s*[:\-]?\s*([0-9,.]+)\s*MTS?', raw, FLAGS)
        # Description block with larger capture and preserving newlines
        desc_after_hs_match = re.search(
            r'HS\s*CODE\s*[:\-]?\s*[0-9\.]{6,10}\s*([\s\S]{0,3000}?)(?:TOTAL\s*NET\s*WEIGHT|TOTAL\s*GROSS\s*WEIGHT|Amount\s*Chargeable|BIN\s*NO|DECLARATION|RATE\s*PER|NO\.?\s*OF|$)',
            raw, FLAGS
        )
        desc_after_hs = desc_after_hs_match.group(1) if desc_after_hs_match else ""
        pre_hs_match = re.search(r'([^\r\n]{8,240})\s*\r?\n\s*HS\s*CODE', raw, FLAGS)
        if not pre_hs_match:
            pre_hs_match = re.search(r'([^\r\n]{8,240})\s*HS\s*CODE', raw, FLAGS)
        pre_hs = pre_hs_match.group(1).strip() if pre_hs_match else ""
        desc_block = (pre_hs + "\n" + desc_after_hs).strip()
        desc_block = re.sub(r'[ \t]{2,}', ' ', desc_block)
        units_guess = (total_net_match.group(1) if total_net_match else find(r'([0-9,.]+)\s*MTS?'))
        amount_guess = find(r'Total\s*:?\s*([0-9,.]+)')
        desc_full = desc_block
        if total_net_wt or total_gross_wt:
            tail = []
            if total_net_wt:
                tail.append(f"TOTAL NET WEIGHT: {total_net_wt} MTS")
            if total_gross_wt:
                tail.append(f"TOTAL GROSS WEIGHT: {total_gross_wt} MTS")
            desc_full = (desc_block + "\n" + "\n".join(tail)).strip()
        # Prepend HS code when available
        hs_code_val = (hs_only.group(1) if hs_only else find(r'HS\s*CODE\s*:?\s*([0-9\.]+)'))
        if hs_code_val:
            desc_full = (f"HS CODE: {hs_code_val}\n" + desc_full).strip()
        # Compose weight & measurements
        wm_lines = []
        if total_net_wt:
            wm_lines.append(f"NET: {total_net_wt} MTS")
        if total_gross_wt:
            wm_lines.append(f"GROSS: {total_gross_wt} MTS")
        if measurement_cbm and measurement_cbm != "NOT FOUND":
            wm_lines.append(f"MEASUREMENT: {measurement_cbm}")
        weight_measurements_fb = "\n".join(wm_lines)
        goods.append({
            "hs_code": hs_code_val,
            "description": desc_full,
            "quantity": "",
            "weight": "",
            "packing": "",
            "unit": "",
            "rate": rate_per_unit if rate_per_unit != "NOT FOUND" else "",
            "amount": amount_guess if amount_guess else "",
            "units_mt": units_guess if units_guess else "",
            "weight_measurements": weight_measurements_fb,
            # include some invoice hints in fallback description
            "description_full_fallback": desc_full,
            "sr_marks": sr_marks_block
        })

    # ✨ RANDOM DEFAULTS FOR MISSING FIELDS
    vessels = ["MSC LORETO", "CMA CGM NEVADA", "APL TOKYO", "MAERSK OHIO", "ONE HAMBURG", "WAN HAI 528", "EVER GIVEN"]
    # Always randomize vessel/voyage per request
    voyage_code = f"V.{random.randint(100,999)}{random.choice(list('ABCDE'))}"
    vessel_voyage = f"{random.choice(vessels)} {voyage_code}"

    container_no = ''.join(random.choices(string.ascii_uppercase, k=4)) + ''.join(random.choices(string.digits, k=7))
    seal_no = ''.join(random.choices(string.digits, k=6))

    # Delivery Agent: random always as requested
    agents = [
        "SEA LINE LOGISTICS PTE. LTD., Singapore",
        "GULF STAR SHIPPING LLC, Dubai",
        "PACIFIC FREIGHT SERVICES, Singapore",
        "BLUE OCEAN LINES, Mumbai",
        "NORTH HARBOUR AGENCIES, Singapore"
    ]
    delivery_agent = random.choice(agents)

    # ✅ STRUCTURED OUTPUT
    # Final fallback: ensure exporter_name is populated if still empty by scanning exporter_block or raw
    if not exporter_name:
        # try first line from exporter_block
        if exporter_block:
            lines = [ln.strip() for ln in re.split(r'[\r\n]+', exporter_block) if ln.strip()]
            for ln in lines:
                if ln and not re.search(r'^(Invoice|PO|I\.?E\.?|Buyer)', ln, re.IGNORECASE):
                    exporter_name = ln
                    break
        # try labeled 'Exporter:' in raw text
        if not exporter_name:
            m = re.search(r'Exporter\s*[:\-]?\s*(.+)', raw, FLAGS)
            if m:
                candidate = m.group(1).splitlines()[0].strip()
                if candidate and not re.search(r'^(Invoice|PO|I\.?E\.?|Buyer)', candidate, re.IGNORECASE):
                    exporter_name = candidate
    return {
        "invoice_no": invoice_no,
        "ie_code": ie_code,
        "po_no": po_no,
        "exporter": exporter_block,
        "exporter_address": exporter_address,
        "exporter_name": exporter_name,
        "consignee": consignee_block,
        "notify_party": notify_block,
        "country_of_origin": country_origin,
        "country_of_final_destination": country_destination,
        "terms": terms,
        "terms_of_payment": terms,
        "drawback_no": drawback_no,
        "benefit_scheme": benefit_scheme,
        "total_amount": total,
        "currency": currency,
        "pre_carriage_by": pre_carriage,
        "vessel_voyage": vessel_voyage,
        "place_of_receipt": place_receipt,
        "place_of_acceptance": (port_loading or place_receipt),
        "port_of_loading": port_loading,
        "port_of_discharge": port_discharge,
        "place_of_delivery": (port_discharge or final_destination),
        "final_destination": final_destination,
        "container_no": container_no,
        "seal_no": seal_no,
        "goods": goods if goods else [],
        "delivery_agent": delivery_agent,
    }

# -------------------------------------------------------
# PDF GENERATION
# -------------------------------------------------------
def generate_bl_pdf(data: dict, template_path="image.jpeg") -> bytes:
    """Overlay extracted data onto Bill of Lading template."""
    buffer = io.BytesIO()
    # Resolve background path safely; fallback if missing
    bg_path = template_path if os.path.isabs(template_path) else os.path.join(os.path.dirname(__file__), template_path)
    c = None
    bg = None
    try:
        if os.path.exists(bg_path):
            bg = Image.open(bg_path)
            w, h = bg.size
            c = canvas.Canvas(buffer, pagesize=(w, h))
            c.drawImage(ImageReader(bg), 0, 0, width=w, height=h)
        else:
            from reportlab.lib.pagesizes import A4
            w, h = A4
            c = canvas.Canvas(buffer, pagesize=A4)
    except Exception:
        from reportlab.lib.pagesizes import A4
        w, h = A4
        c = canvas.Canvas(buffer, pagesize=A4)
    finally:
        if bg is not None:
            try:
                bg.close()
            except Exception:
                pass
    c.setFont("Helvetica", 9)

    def draw_wrapped(text, x, y, max_width):
        if not text: return
        # Support multi-paragraph text (\n separated)
        paragraphs = re.split(r"\r?\n", text)
        line_offset = 0
        for para in paragraphs:
            if not para:
                line_offset += 11
                continue
            words, lines, line = para.split(), [], ""
            for word in words:
                test = f"{line}{word} "
                if c.stringWidth(test, "Helvetica", 9) < max_width:
                    line = test
                else:
                    lines.append(line.strip())
                    line = f"{word} "
            lines.append(line.strip())
            for l in lines:
                c.drawString(x, y - line_offset, l)
                line_offset += 11

    def draw_wrapped_box(text, x, y, max_width, max_lines=None, align='left', font='Helvetica', font_size=9):
        """Draw wrapped text in a box. Returns height used in points.
        align='left' or 'right' (for right, x is the right-edge coordinate).
        """
        if not text:
            return 0
        c.setFont(font, font_size)
        paragraphs = re.split(r"\r?\n", text)
        lines = []
        for para in paragraphs:
            if not para:
                lines.append("")
                continue
            words = para.split()
            line = ""
            for word in words:
                test = f"{line}{word} "
                if c.stringWidth(test, font, font_size) < max_width:
                    line = test
                else:
                    lines.append(line.strip())
                    line = f"{word} "
            if line:
                lines.append(line.strip())
        if max_lines is not None:
            lines = lines[:max_lines]
        line_height = font_size + 2
        for i, l in enumerate(lines):
            yy = y - (i * line_height)
            if align == 'left':
                c.drawString(x, yy, l)
            else:
                # x is right edge for right-aligned text
                c.drawRightString(x, yy, l)
        return len(lines) * line_height

    # SHIPPER
    c.setFont("Helvetica-Bold", 10)
    c.drawString(70, h - 72, " SHIPPER:")
    c.setFont("Helvetica", 9)
    # Prepare exporter display values and sanitize placeholder labels
    exp_name = (data.get("exporter_name") or "").strip()
    exp_block = (data.get("exporter") or "").strip()
    exp_address = (data.get("exporter_address") or "").strip()
    # Remove invoice/PO/IE Code lines from exporter block so they don't appear in the exporter box
    if exp_block:
        exp_block = re.sub(r'(?mi)^\s*Invoice\s*No\.?\s*[:\-]?.*$\n?', '', exp_block)
        exp_block = re.sub(r'(?mi)^\s*Buyer\'?s\s*Order\s*No\.?\s*[:\-]?.*$\n?', '', exp_block)
        exp_block = re.sub(r'(?mi)^\s*I\.?E\.?\s*Code\.?\s*[:\-]?.*$\n?', '', exp_block)
        exp_block = re.sub(r'(?mi)^\s*I\.E\.?\s*Code\.?\s*[:\-]?.*$\n?', '', exp_block)
        # Remove leading 'Exporter:' or 'Shipper:' labels if present so splitting yields name + address
        exp_block = re.sub(r'(?mi)^(?:Exporter|Shipper)\s*:?\s*', '', exp_block)
        exp_block = exp_block.strip()
    # remove accidental literal headings
    exp_name = re.sub(r'(?i)^Exporter\s*:?\s*$', '', exp_name).strip()
    # If no clean name, try to pick first non-heading line from exporter block
    if not exp_name:
        lines = [ln.strip() for ln in re.split(r'[\r\n]+', exp_block) if ln.strip()]
        for ln in lines:
            if not re.search(r'^(INVOICE|EXPORTER|SHIPPER|BILL|DATE|TAX)$', ln, re.IGNORECASE):
                exp_name = ln
                break
        if not exp_name and lines:
            exp_name = lines[0]
        # set address from remaining lines if not already set
        if lines and not exp_address:
            try:
                idx = lines.index(exp_name)
                exp_address = "\n".join(lines[idx+1:]).strip()
            except Exception:
                exp_address = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    # Draw exporter name (bold) then address (regular)
    # Sanitize exporter name/address: remove invoice, PO, I.E. Code lines if present
    def remove_invoice_po_ie(text: str) -> str:
        if not text:
            return ""
        lines = [ln for ln in re.split(r'[\r\n]+', text) if ln.strip()]
        filtered = []
        for ln in lines:
            if re.search(r'Invoice\s*No\.?|Buyer\'?s\s*Order|I\.?E\.?\s*Code|I\.E\.\s*Code', ln, re.IGNORECASE):
                continue
            filtered.append(ln)
        return "\n".join(filtered).strip()

    exp_name = remove_invoice_po_ie(exp_name)
    exp_address = remove_invoice_po_ie(exp_address)

    exp_y_top = h - 84  # slightly higher (move exporter box up)
    # If exporter name missing but we have an exporter_address or exporter block,
    # try to derive a company-like name so SHIPPER shows both name and address.
    if not exp_name:
        company_token = re.compile(r'\b(IMPEX|PVT|LTD|LLP|LLC|TRADING|EXPORTS|IMPORTS|CO\.|COMPANY|INDIA)\b', re.IGNORECASE)
        # Try to extract from the explicit exporter_address returned by extractor
        if exp_address:
            addr_lines = [ln.strip() for ln in re.split(r'[\r\n]+', exp_address) if ln.strip()]
            candidate = None
            for ln in addr_lines:
                # prefer lines that look like a company
                if company_token.search(ln) or re.match(r'^[A-Z][A-Za-z0-9&\-,\. ]{2,}$', ln):
                    candidate = ln
                    break
            # fallback: title-case or short first line
            if not candidate and addr_lines:
                for ln in addr_lines:
                    if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)+', ln):
                        candidate = ln
                        break
            if candidate:
                # remove candidate from address and set as name
                exp_name = candidate
                try:
                    # remove only the first occurrence
                    idx = addr_lines.index(candidate)
                    remaining = addr_lines[:idx] + addr_lines[idx+1:]
                    exp_address = '\n'.join(remaining).strip()
                except ValueError:
                    # safest fallback
                    exp_address = '\n'.join([l for l in addr_lines if l != candidate]).strip()
        # If still not found, try to use the raw exporter block
        if not exp_name and exp_block:
            lines = [ln.strip() for ln in re.split(r'[\r\n]+', exp_block) if ln.strip()]
            if lines:
                exp_name = lines[0]
                exp_address = exp_address or '\n'.join(lines[1:]).strip()
    # Final aggressive fallback: scan other data fields for a company-like name (sometimes extractor puts name elsewhere)
    if not exp_name:
        company_token = re.compile(r'\b(IMPEX|PVT|LTD|LLP|LLC|TRADING|EXPORTS|IMPORTS|CO\.|COMPANY|SHARDHA|SHRADHA|IMPEX|EXPORTS|INDIA)\b', re.IGNORECASE)
        candidate = None
        # Search through string fields in data (consignee/notify/etc.) for a likely exporter name
        for k, v in data.items():
            if not v or not isinstance(v, str):
                continue
            # skip fields that are obviously consignee/notify or vessel info
            if k in ('consignee', 'notify_party'):
                continue
            for ln in [l.strip() for l in re.split(r'[\r\n]+', v) if l.strip()]:
                # skip generic headings
                if re.search(r'^(INVOICE|BILL|TAX|DATE|PLACE|PORT)\b', ln, re.IGNORECASE):
                    continue
                if company_token.search(ln) or re.match(r'^[A-Z][A-Z0-9 &,\-\.]{2,}$', ln):
                    candidate = ln
                    break
            if candidate:
                break
        if candidate:
            exp_name = candidate
            # remove candidate line from exporter block/address if present
            if candidate and exp_address and candidate in exp_address:
                exp_address = '\n'.join([l for l in re.split(r'[\r\n]+', exp_address) if l.strip() and l.strip() != candidate]).strip()
    if exp_name:
        c.setFont("Helvetica-Bold", 9)
        draw_wrapped(exp_name, 70, exp_y_top, 350)
    if exp_address:
        c.setFont("Helvetica", 9)
        draw_wrapped(exp_address, 70, exp_y_top - 18, 350)

    # CONSIGNEE
    c.setFont("Helvetica-Bold", 10)
    c.drawString(70, h - 200, "CONSIGNEE:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("consignee", ""), 70, h - 220, 350)

    # NOTIFY PARTY
    c.setFont("Helvetica-Bold", 10)
    c.drawString(70, h - 300, "NOTIFY PARTY:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("notify_party", ""), 70, h - 320, 350)

    # PORTS
    # PLACE OF ACCEPTANCE
    c.setFont("Helvetica-Bold", 10)
    c.drawString(70, h - 460, "PLACE OF ACCEPTANCE:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("port_of_loading", ""), 70, h - 480, 350)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(460, h - 460, "PORT OF LOADING:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("port_of_loading", ""), 460, h - 480, 350)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(70, h - 510, "PORT OF DISCHARGE:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("port_of_discharge", ""), 70, h - 530, 350)
    # Final Destination value suppressed in header area (per request)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(460, h - 510, "PLACE OF DELIVERY:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("port_of_discharge", ""), 460, h - 530, 350)

    # Removed white rectangles as requested

    # VESSEL / B/L NO.
    c.setFont("Helvetica-Bold", 10)
    c.drawString(70, h - 410, "VESSEL/VOYAGE:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("vessel_voyage", ""), 200, h - 410, 400)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(450, h - 390, f"B/L NO.: {data.get('invoice_no', '')}")

    # GOODS table boxes mapping (tuned positions)
    left_box_x = 100          # fine tune columns
    desc_box_x = 330
    right_box_x = max(w - 200, 520)
    y_start = h - 590
    # If container/seal present, draw FCL token and the container heading + numbers once at the top-left
 # Extract FCL count and prepare container/seal data
    c_no = data.get('container_no', '')
    s_no = data.get('seal_no', '')
    fcl_token = ''
    fcl_count = 6  # default to 1 container
    goods_list_for_fcl = data.get('goods', [])
    if goods_list_for_fcl:
        possible_sr = goods_list_for_fcl[0].get('sr_marks', '') or ''
        m_fcl = re.search(r"(\d+\s*X\s*20[\'\’\″\”\"]?\s*FCL)", possible_sr, re.IGNORECASE)
        if m_fcl:
            fcl_token = m_fcl.group(1)
    # Draw FCL token once at the top
    if fcl_token or (c_no and s_no):
        top_container_y = y_start + 10
        line_h = 11
        c.setFont("Helvetica", 9)
        if fcl_token:
            draw_wrapped(fcl_token, left_box_x, top_container_y, 200)
            top_container_y -= line_h
        
        # Draw container & seal information multiple times based on FCL count
        if c_no and s_no:
            c.setFont("Helvetica-Bold", 8)
            draw_wrapped("Container & Seal nos.:", left_box_x, top_container_y, 200)
            top_container_y -= line_h
            c.setFont("Helvetica", 8)
            
            # Repeat container/seal numbers based on FCL count
            for i in range(fcl_count):
                # Generate unique container and seal numbers for each
                if i == 0:
                    # Use the original numbers for the first one
                    container = c_no
                    seal = s_no
                else:
                    # Generate new random numbers for additional containers
                    container = ''.join(random.choices(string.ascii_uppercase, k=4)) + ''.join(random.choices(string.digits, k=7))
                    seal = ''.join(random.choices(string.digits, k=6))
                
                draw_wrapped(f"{container} / {seal}", left_box_x, top_container_y, 200)
                top_container_y -= line_h

    for i, good in enumerate(data.get("goods", [])):
        # push first row down slightly to avoid overlap with top container line
        row_y = y_start - (i * 115) - (20 if (i == 0 and (fcl_token or (c_no and s_no))) else 0)
        c.setFont("Helvetica", 9)
        # Sr No & Marks – left column (remove any container/seal fragments and FCL tokens)
        sr_text = good.get('sr_marks') or ''
        if sr_text:
            # remove container & seal and FCL occurrences to avoid duplication
            sr_clean = re.sub(r'Container\s*&\s*Seal\s*nos?\.?[:\-]?\s*.*', '', sr_text, flags=re.IGNORECASE).strip()
            sr_clean = re.sub(r"(\d+\s*X\s*20[\'\’\″\”\"]?\s*FCL)", '', sr_clean, flags=re.IGNORECASE).strip()
            if sr_clean:
                draw_wrapped(sr_clean, left_box_x, row_y, 200)
        # Description of Goods – middle column (preserve pre-HS line above HS CODE)
        desc = good.get('description','') or ''
        # Normalize newlines and split
        desc_lines = [ln.strip() for ln in re.split(r'\r?\n', desc) if ln.strip()]
    # remove container & seal lines that sometimes appear in the description column
        desc_lines = [ln for ln in desc_lines if not re.search(r'Container\s*&\s*Seal|Container\s&\sSeal|Container\s*&\s*Seal\snos?', ln, re.IGNORECASE)]
        # If description contains HS CODE as a line, try to move any product-name lines above it
        hs_idx = None
        for idx, ln in enumerate(desc_lines):
            if re.search(r'HS\s*CODE', ln, re.IGNORECASE):
                hs_idx = idx
                break
        ordered_desc = []
        if hs_idx is not None:
            # lines before HS -> product name(s)
            ordered_desc.extend(desc_lines[:hs_idx])
            ordered_desc.append(desc_lines[hs_idx])
            ordered_desc.extend(desc_lines[hs_idx+1:])
        else:
            ordered_desc = desc_lines
        draw_wrapped('\n'.join(ordered_desc), desc_box_x, row_y, 430)
        # Numeric columns – units and rate only (no amount in this column)
        units_x = right_box_x
        rate_x = units_x + 100
    
        draw_wrapped(str(good.get('units_mt','')), units_x, row_y, 80)
        draw_wrapped(str(good.get('rate','')), rate_x, row_y, 80)
        # Weight & Measurements details in the far-right narrow column
        wm = good.get('weight_measurements','')
        if wm:
            wm_x = max(w - 155, rate_x + 120)
            draw_wrapped(wm, wm_x, row_y, 140)

    # Footer box: Delivery Agent (left) and Place & Date (right) – placed inside footer box with better spacing
    footer_y = 190  # slightly higher so headings and data fit neatly inside footer box
    # Delivery Agent: heading above the agent block
    heading_font = "Helvetica-Bold"
    heading_size = 10
    data_font = "Helvetica"
    data_size = 9
    # Draw heading (above) then data (below) with wrapping limits
    heading_y = footer_y + 20
    # place data directly beneath the heading with a small gap
    data_y = heading_y - 14
    c.setFont(heading_font, heading_size)
    c.drawString(70, heading_y, "DELIVERY AGENT:")
    c.setFont(data_font, data_size)
    # left area: start at x=200, width ~420, limit to 3 lines to avoid overflow
    draw_wrapped_box(data.get("delivery_agent", ""), 150, data_y, 380, max_lines=3, align='left', font=data_font, font_size=data_size)

    from datetime import datetime
    place = (data.get("port_of_loading") or data.get("place_of_receipt") or "").strip()
    today = datetime.now().strftime("%d-%m-%Y")
    place_date_text = f"{place}  {today}" if place else today

    # Place & Date: heading above the date block on the right
    c.setFont(heading_font, heading_size)
    c.drawRightString(w - 70, heading_y, "PLACE & DATE:")
    c.setFont(data_font, data_size)
    # right-edge is w - 70; limit to 2 lines and align right
    draw_wrapped_box(place_date_text, w - 70, data_y, 180, max_lines=2, align='right', font=data_font, font_size=data_size)

    c.save()
    return buffer.getvalue()




def generate_checklist_pdf(data: dict, template_path="checklist.png") -> bytes:
    """Overlay extracted data onto Bill of Lading template."""
    buffer = io.BytesIO()
    # Resolve background path safely; fallback if missing
    bg_path = template_path if os.path.isabs(template_path) else os.path.join(os.path.dirname(__file__), template_path)
    c = None
    bg = None
    try:
        if os.path.exists(bg_path):
            bg = Image.open(bg_path)
            w, h = bg.size
            c = canvas.Canvas(buffer, pagesize=(w, h))
            c.drawImage(ImageReader(bg), 0, 0, width=w, height=h)
        else:
            from reportlab.lib.pagesizes import A4
            w, h = A4
            c = canvas.Canvas(buffer, pagesize=A4)
    except Exception:
        from reportlab.lib.pagesizes import A4
        w, h = A4
        c = canvas.Canvas(buffer, pagesize=A4)
    finally:
        if bg is not None:
            try:
                bg.close()
            except Exception:
                pass
    c.setFont("Helvetica", 9)

    def draw_wrapped(text, x, y, max_width):
        if not text: return
        # Support multi-paragraph text (\n separated)
        paragraphs = re.split(r"\r?\n", text)
        line_offset = 0
        for para in paragraphs:
            if not para:
                line_offset += 11
                continue
            words, lines, line = para.split(), [], ""
            for word in words:
                test = f"{line}{word} "
                if c.stringWidth(test, "Helvetica", 9) < max_width:
                    line = test
                else:
                    lines.append(line.strip())
                    line = f"{word} "
            lines.append(line.strip())
            for l in lines:
                c.drawString(x, y - line_offset, l)
                line_offset += 11

    def draw_wrapped_box(text, x, y, max_width, max_lines=None, align='left', font='Helvetica', font_size=9):
        """Draw wrapped text in a box. Returns height used in points.
        align='left' or 'right' (for right, x is the right-edge coordinate).
        """
        if not text:
            return 0
        c.setFont(font, font_size)
        paragraphs = re.split(r"\r?\n", text)
        lines = []
        for para in paragraphs:
            if not para:
                lines.append("")
                continue
            words = para.split()
            line = ""
            for word in words:
                test = f"{line}{word} "
                if c.stringWidth(test, font, font_size) < max_width:
                    line = test
                else:
                    lines.append(line.strip())
                    line = f"{word} "
            if line:
                lines.append(line.strip())
        if max_lines is not None:
            lines = lines[:max_lines]
        line_height = font_size + 2
        for i, l in enumerate(lines):
            yy = y - (i * line_height)
            if align == 'left':
                c.drawString(x, yy, l)
            else:
                # x is right edge for right-aligned text
                c.drawRightString(x, yy, l)
        return len(lines) * line_height

    # SHIPPER
    c.setFont("Helvetica-Bold", 10)
    # c.drawString(70, h - 72, " SHIPPER:")
    c.setFont("Helvetica", 9)
    # Prepare exporter display values and sanitize placeholder labels
    exp_name = (data.get("exporter_name") or "").strip()
    exp_block = (data.get("exporter") or "").strip()
    exp_address = (data.get("exporter_address") or "").strip()
    # Remove invoice/PO/IE Code lines from exporter block so they don't appear in the exporter box
    if exp_block:
        exp_block = re.sub(r'(?mi)^\s*Invoice\s*No\.?\s*[:\-]?.*$\n?', '', exp_block)
        exp_block = re.sub(r'(?mi)^\s*Buyer\'?s\s*Order\s*No\.?\s*[:\-]?.*$\n?', '', exp_block)
        exp_block = re.sub(r'(?mi)^\s*I\.?E\.?\s*Code\.?\s*[:\-]?.*$\n?', '', exp_block)
        exp_block = re.sub(r'(?mi)^\s*I\.E\.?\s*Code\.?\s*[:\-]?.*$\n?', '', exp_block)
        # Remove leading 'Exporter:' or 'Shipper:' labels if present so splitting yields name + address
        exp_block = re.sub(r'(?mi)^(?:Exporter|Shipper)\s*:?\s*', '', exp_block)
        exp_block = exp_block.strip()
    # remove accidental literal headings
    exp_name = re.sub(r'(?i)^Exporter\s*:?\s*$', '', exp_name).strip()
    # If no clean name, try to pick first non-heading line from exporter block
    if not exp_name:
        lines = [ln.strip() for ln in re.split(r'[\r\n]+', exp_block) if ln.strip()]
        for ln in lines:
            if not re.search(r'^(INVOICE|EXPORTER|SHIPPER|BILL|DATE|TAX)$', ln, re.IGNORECASE):
                exp_name = ln
                break
        if not exp_name and lines:
            exp_name = lines[0]
        # set address from remaining lines if not already set
        if lines and not exp_address:
            try:
                idx = lines.index(exp_name)
                exp_address = "\n".join(lines[idx+1:]).strip()
            except Exception:
                exp_address = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

    # Draw exporter name (bold) then address (regular)
    # Sanitize exporter name/address: remove invoice, PO, I.E. Code lines if present
    def remove_invoice_po_ie(text: str) -> str:
        if not text:
            return ""
        lines = [ln for ln in re.split(r'[\r\n]+', text) if ln.strip()]
        filtered = []
        for ln in lines:
            if re.search(r'Invoice\s*No\.?|Buyer\'?s\s*Order|I\.?E\.?\s*Code|I\.E\.\s*Code', ln, re.IGNORECASE):
                continue
            filtered.append(ln)
        return "\n".join(filtered).strip()

    exp_name = remove_invoice_po_ie(exp_name)
    exp_address = remove_invoice_po_ie(exp_address)

    exp_y_top = h - 300  # slightly higher (move exporter box up)
    # If exporter name missing but we have an exporter_address or exporter block,
    # try to derive a company-like name so SHIPPER shows both name and address.
    if not exp_name:
        company_token = re.compile(r'\b(IMPEX|PVT|LTD|LLP|LLC|TRADING|EXPORTS|IMPORTS|CO\.|COMPANY|INDIA)\b', re.IGNORECASE)
        # Try to extract from the explicit exporter_address returned by extractor
        if exp_address:
            addr_lines = [ln.strip() for ln in re.split(r'[\r\n]+', exp_address) if ln.strip()]
            candidate = None
            for ln in addr_lines:
                # prefer lines that look like a company
                if company_token.search(ln) or re.match(r'^[A-Z][A-Za-z0-9&\-,\. ]{2,}$', ln):
                    candidate = ln
                    break
            # fallback: title-case or short first line
            if not candidate and addr_lines:
                for ln in addr_lines:
                    if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)+', ln):
                        candidate = ln
                        break
            if candidate:
                # remove candidate from address and set as name
                exp_name = candidate
                try:
                    # remove only the first occurrence
                    idx = addr_lines.index(candidate)
                    remaining = addr_lines[:idx] + addr_lines[idx+1:]
                    exp_address = '\n'.join(remaining).strip()
                except ValueError:
                    # safest fallback
                    exp_address = '\n'.join([l for l in addr_lines if l != candidate]).strip()
        # If still not found, try to use the raw exporter block
        if not exp_name and exp_block:
            lines = [ln.strip() for ln in re.split(r'[\r\n]+', exp_block) if ln.strip()]
            if lines:
                exp_name = lines[0]
                exp_address = exp_address or '\n'.join(lines[1:]).strip()
    # Final aggressive fallback: scan other data fields for a company-like name (sometimes extractor puts name elsewhere)
    if not exp_name:
        company_token = re.compile(r'\b(IMPEX|PVT|LTD|LLP|LLC|TRADING|EXPORTS|IMPORTS|CO\.|COMPANY|SHARDHA|SHRADHA|IMPEX|EXPORTS|INDIA)\b', re.IGNORECASE)
        candidate = None
        # Search through string fields in data (consignee/notify/etc.) for a likely exporter name
        for k, v in data.items():
            if not v or not isinstance(v, str):
                continue
            # skip fields that are obviously consignee/notify or vessel info
            if k in ('consignee', 'notify_party'):
                continue
            for ln in [l.strip() for l in re.split(r'[\r\n]+', v) if l.strip()]:
                # skip generic headings
                if re.search(r'^(INVOICE|BILL|TAX|DATE|PLACE|PORT)\b', ln, re.IGNORECASE):
                    continue
                if company_token.search(ln) or re.match(r'^[A-Z][A-Z0-9 &,\-\.]{2,}$', ln):
                    candidate = ln
                    break
            if candidate:
                break
        if candidate:
            exp_name = candidate
            # remove candidate line from exporter block/address if present
            if candidate and exp_address and candidate in exp_address:
                exp_address = '\n'.join([l for l in re.split(r'[\r\n]+', exp_address) if l.strip() and l.strip() != candidate]).strip()
    if exp_name:
        c.setFont("Helvetica-Bold", 9)
        draw_wrapped(exp_name, 70, exp_y_top, 350)
    if exp_address:
        c.setFont("Helvetica", 9)
        draw_wrapped(exp_address, 70, exp_y_top - 18, 350)

    # CONSIGNEE
    c.setFont("Helvetica-Bold", 10)
    # c.drawString(720, h - 300, "CONSIGNEE:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("consignee", ""), 720, h - 320, 350)

    # NOTIFY PARTY
    c.setFont("Helvetica-Bold", 10)
    # c.drawString(740, h - 180, "NOTIFY PARTY:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("notify_party", ""), 740, h - 200, 350)

    # PORTS
    # PLACE OF ACCEPTANCE
    c.setFont("Helvetica-Bold", 10)
    # c.drawString(210, h - 570, "PLACE OF ACCEPTANCE:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("port_of_loading", ""), 210, h - 585, 350)

    # c.setFont("Helvetica-Bold", 10)
    # c.drawString(240, h - 440, "PORT OF LOADING:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("port_of_loading", ""), 240, h - 455, 350)

    # c.setFont("Helvetica-Bold", 10)
    # c.drawString(240, h - 480, "PORT OF DISCHARGE:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("port_of_discharge", ""), 240, h - 490, 350)
    # Final Destination value suppressed in header area (per request)

    # c.setFont("Helvetica-Bold", 10)
    # c.drawString(250, h - 500, "PLACE OF DELIVERY:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("port_of_discharge", ""), 250, h - 520, 350)

    # Removed white rectangles as requested

    # VESSEL / B/L NO.
    c.setFont("Helvetica-Bold", 10)
    c.drawString(70, h - 410, "VESSEL/VOYAGE:")
    c.setFont("Helvetica", 9)
    draw_wrapped(data.get("vessel_voyage", ""), 200, h - 410, 400)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(450, h - 390, f"B/L NO.: {data.get('invoice_no', '')}")

    # GOODS table boxes mapping (tuned positions)
    left_box_x = 50          # fine tune columns
    desc_box_x = 620
    right_box_x = max(w - 200, 520)
    y_start = h - 1450
    # If container/seal present, draw FCL token and the container heading + numbers once at the top-left
 # Extract FCL count and prepare container/seal data
    c_no = data.get('container_no', '')
    s_no = data.get('seal_no', '')
    fcl_token = ''
    fcl_count = 6  # default to 1 container
    goods_list_for_fcl = data.get('goods', [])
    if goods_list_for_fcl:
        possible_sr = goods_list_for_fcl[0].get('sr_marks', '') or ''
        m_fcl = re.search(r"(\d+\s*X\s*20[\'\’\″\”\"]?\s*FCL)", possible_sr, re.IGNORECASE)
        if m_fcl:
            fcl_token = m_fcl.group(1)
    # Draw FCL token once at the top
    if fcl_token or (c_no and s_no):
        top_container_y = y_start + 10
        line_h = 11
        c.setFont("Helvetica", 9)
        if fcl_token:
            draw_wrapped(fcl_token, left_box_x, top_container_y, 200)
            top_container_y -= line_h
        
        # Draw container & seal information multiple times based on FCL count
        if c_no and s_no:
            c.setFont("Helvetica-Bold", 8)
            draw_wrapped("Container & Seal nos.:", left_box_x, top_container_y, 200)
            top_container_y -= line_h
            c.setFont("Helvetica", 8)
            
            # Repeat container/seal numbers based on FCL count
            for i in range(fcl_count):
                # Generate unique container and seal numbers for each
                if i == 0:
                    # Use the original numbers for the first one
                    container = c_no
                    seal = s_no
                else:
                    # Generate new random numbers for additional containers
                    container = ''.join(random.choices(string.ascii_uppercase, k=4)) + ''.join(random.choices(string.digits, k=7))
                    seal = ''.join(random.choices(string.digits, k=6))
                
                draw_wrapped(f"{container} / {seal}", left_box_x, top_container_y, 200)
                top_container_y -= line_h

    for i, good in enumerate(data.get("goods", [])):
        # push first row down slightly to avoid overlap with top container line
        row_y = y_start - (i * 115) - (20 if (i == 0 and (fcl_token or (c_no and s_no))) else 0)
        c.setFont("Helvetica", 9)
        # Sr No & Marks – left column (remove any container/seal fragments and FCL tokens)
        sr_text = good.get('sr_marks') or ''
        if sr_text:
            # remove container & seal and FCL occurrences to avoid duplication
            sr_clean = re.sub(r'Container\s*&\s*Seal\s*nos?\.?[:\-]?\s*.*', '', sr_text, flags=re.IGNORECASE).strip()
            sr_clean = re.sub(r"(\d+\s*X\s*20[\'\’\″\”\"]?\s*FCL)", '', sr_clean, flags=re.IGNORECASE).strip()
            if sr_clean:
                draw_wrapped(sr_clean, left_box_x, row_y, 200)
        # Description of Goods – middle column (preserve pre-HS line above HS CODE)
        desc = good.get('description','') or ''
        # Normalize newlines and split
        desc_lines = [ln.strip() for ln in re.split(r'\r?\n', desc) if ln.strip()]
    # remove container & seal lines that sometimes appear in the description column
        desc_lines = [ln for ln in desc_lines if not re.search(r'Container\s*&\s*Seal|Container\s&\sSeal|Container\s*&\s*Seal\snos?', ln, re.IGNORECASE)]
        # If description contains HS CODE as a line, try to move any product-name lines above it
        hs_idx = None
        for idx, ln in enumerate(desc_lines):
            if re.search(r'HS\s*CODE', ln, re.IGNORECASE):
                hs_idx = idx
                break
        ordered_desc = []
        if hs_idx is not None:
            # lines before HS -> product name(s)
            ordered_desc.extend(desc_lines[:hs_idx])
            ordered_desc.append(desc_lines[hs_idx])
            ordered_desc.extend(desc_lines[hs_idx+1:])
        else:
            ordered_desc = desc_lines
        draw_wrapped('\n'.join(ordered_desc), desc_box_x, row_y, 430)
        # Numeric columns – units and rate only (no amount in this column)
        units_x = right_box_x
        rate_x = units_x + 100
    
        draw_wrapped(str(good.get('units_mt','')), units_x, row_y, 80)
        draw_wrapped(str(good.get('rate','')), rate_x, row_y, 80)
        # Weight & Measurements details in the far-right narrow column
        wm = good.get('weight_measurements','')
        if wm:
            wm_x = max(w - 155, rate_x + 120)
            draw_wrapped(wm, wm_x, row_y, 140)

    # Footer box: Delivery Agent (left) and Place & Date (right) – placed inside footer box with better spacing
    footer_y = 190  # slightly higher so headings and data fit neatly inside footer box
    # Delivery Agent: heading above the agent block
    heading_font = "Helvetica-Bold"
    heading_size = 10
    data_font = "Helvetica"
    data_size = 9
    # Draw heading (above) then data (below) with wrapping limits
    heading_y = footer_y + 20
    # place data directly beneath the heading with a small gap
    data_y = heading_y - 14
    c.setFont(heading_font, heading_size)
    c.drawString(70, heading_y, "DELIVERY AGENT:")
    c.setFont(data_font, data_size)
    # left area: start at x=200, width ~420, limit to 3 lines to avoid overflow
    draw_wrapped_box(data.get("delivery_agent", ""), 150, data_y, 380, max_lines=3, align='left', font=data_font, font_size=data_size)

    from datetime import datetime
    place = (data.get("port_of_loading") or data.get("place_of_receipt") or "").strip()
    today = datetime.now().strftime("%d-%m-%Y")
    place_date_text = f"{place}  {today}" if place else today

    # Place & Date: heading above the date block on the right
    c.setFont(heading_font, heading_size)
    c.drawRightString(w - 70, heading_y, "PLACE & DATE:")
    c.setFont(data_font, data_size)
    # right-edge is w - 70; limit to 2 lines and align right
    draw_wrapped_box(place_date_text, w - 70, data_y, 180, max_lines=2, align='right', font=data_font, font_size=data_size)

    c.save()
    return buffer.getvalue()



# -------------------------------------------------------
# API ROUTES
# -------------------------------------------------------
@app.post("/generate-bl/")
async def generate_bl(invoice_pdf: UploadFile = File(...)):
    if not invoice_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    pdf_bytes = await invoice_pdf.read()
    text = extract_text_from_pdf(pdf_bytes)
    if not text:
        raise HTTPException(422, "No readable text found in PDF")

    data = extract_invoice_data(text)
    
    # --- YOUR ORIGINAL DEBUG LOGIC ---
    print(f"DEBUG - Port of Loading: '{data.get('port_of_loading', 'NOT_FOUND')}'")
    print(f"DEBUG - Raw text sample: '{text[:500]}'")
    # --- END DEBUG LOGIC ---

    # 1. Generate both PDF files
    bl_pdf_content = generate_bl_pdf(data, "image.jpeg")
    checklist_pdf_content = generate_checklist_pdf(data, "checklist.png")

    invoice_no = data.get('invoice_no', 'Unknown')
    
    # 2. Create an in-memory byte buffer for the ZIP file
    zip_buffer = io.BytesIO()

    # 3. Create a ZipFile object and add the PDFs
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # File 1: Bill of Lading
        bl_filename = f"BL_{invoice_no}.pdf"
        zf.writestr(bl_filename, bl_pdf_content)
        
        # File 2: Checklist
        checklist_filename = f"Checklist_{invoice_no}.pdf"
        zf.writestr(checklist_filename, checklist_pdf_content)

    # 4. Rewind the buffer to the beginning
    zip_buffer.seek(0)
    
    # 5. Return the Response with the ZIP file content
    return Response(
        content=zip_buffer.read(),
        media_type="application/zip", # CRITICAL: Must be application/zip
        headers={
            # CRITICAL: Filename must end with .zip
            "Content-Disposition": f"attachment; filename=Documents_{invoice_no}.zip" 
        }
    )
@app.post("/generate-bl-json/")
async def generate_bl_json(invoice_pdf: UploadFile = File(...)):
    if not invoice_pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    pdf_bytes = await invoice_pdf.read()
    text = extract_text_from_pdf(pdf_bytes)
    if not text:
        raise HTTPException(422, "No readable text found in PDF")

    data = extract_invoice_data(text)
    return JSONResponse(content=data)

@app.get("/")
def home():
    return {"message": "Upload your invoice PDF to /generate-bl or /generate-bl-json"}
    