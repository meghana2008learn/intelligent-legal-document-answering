"""
Helper script to generate a realistic 3-page sample legal contract in PDF format
for testing the Intelligent Legal Document Answering application.
"""
import os

def create_sample_pdf(output_path: str):
    # Content for 3 realistic pages of a legal agreement
    page1_lines = [
        "STANDARD MASTER SERVICES AND NON-DISCLOSURE AGREEMENT",
        "Document Reference: AGY-LEGAL-2026-001",
        "",
        "This Agreement is entered into on this 1st day of January, 2026, by and between:",
        "Party A: Apex Digital Solutions Inc., a Delaware corporation ('Company'), and",
        "Party B: Quantum Logistics LLC, a California limited liability company ('Contractor').",
        "",
        "RECITALS:",
        "WHEREAS, Company wishes to engage Contractor for cloud infrastructure consulting;",
        "WHEREAS, Contractor possesses specialized engineering and legal compliance expertise;",
        "NOW, THEREFORE, the parties mutually covenant and agree as follows:",
        "",
        "SECTION 1: PURPOSE AND SCOPE OF AGREEMENT",
        "1.1 Scope of Services: Contractor agrees to provide cloud architecture auditing, security reviews,",
        "and data compliance monitoring as outlined in Statement of Work (SOW) Exhibit A.",
        "1.2 Professional Standards: All services shall be performed in accordance with highest",
        "industry standards and applicable data protection regulations.",
        "1.3 Independent Contractor Status: The relationship between the parties is that of independent",
        "contractors. Nothing herein shall create an employer-employee or agency relationship."
    ]

    page2_lines = [
        "SECTION 2: RESPONSIBILITIES AND IMPORTANT OBLIGATIONS",
        "2.1 Company Responsibilities: Company shall provide Contractor with timely access to technical",
        "specifications, project repositories, and necessary staging cloud environments.",
        "2.2 Contractor Obligations: Contractor shall maintain strict records, deliver weekly audit milestones,",
        "and immediately notify Company of any discovered security vulnerabilities within 24 hours.",
        "",
        "SECTION 3: CONFIDENTIALITY AND NON-DISCLOSURE",
        "3.1 Confidential Information: Means all non-public technical, business, financial, or legal",
        "information disclosed by either party during the term of this Agreement.",
        "3.2 Non-Disclosure Obligation: Each party agrees to hold all Confidential Information in strict",
        "confidence and not disclose it to third parties for a period of five (5) years following termination.",
        "",
        "SECTION 4: TERM AND DURATION OF AGREEMENT",
        "4.1 Duration: This Agreement shall commence on January 1, 2026, and shall remain in full force",
        "and effect for a duration of twenty-four (24) months until December 31, 2027, unless terminated earlier.",
        "4.2 Renewal: The Agreement may be renewed for successive one-year terms upon mutual written consent."
    ]

    page3_lines = [
        "SECTION 5: TERMINATION CONDITIONS",
        "5.1 Termination for Convenience: Either party may terminate this Agreement without cause by giving",
        "at least thirty (30) days prior written notice to the other party.",
        "5.2 Termination for Cause: If either party commits a material breach of any provision,",
        "the non-breaching party may terminate immediately if such breach is not cured within fifteen (15) days.",
        "",
        "SECTION 6: BREACH AND PENALTIES",
        "6.1 Consequences of Breach: In the event of an uncured material breach, the breaching party",
        "shall be liable for direct damages arising out of the failure to perform.",
        "6.2 Liquidated Damages and Penalties: If Contractor breaches Section 3 (Confidentiality) or unauthorizedly",
        "discloses proprietary source code, Contractor shall pay a liquidated damages penalty of $50,000 per violation.",
        "6.3 Late Payment Penalty: Overdue invoices shall accrue interest at the rate of 1.5% per month.",
        "",
        "SECTION 7: GOVERNING LAW AND DISPUTE RESOLUTION",
        "7.1 Governing Law: This Agreement shall be governed by and construed under the laws of the State of Delaware.",
        "7.2 Arbitration: Any dispute arising out of this Agreement shall be settled by binding arbitration in Wilmington."
    ]

    pages = [page1_lines, page2_lines, page3_lines]

    def escape_text(text: str) -> str:
        return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    # Build objects
    objects = []
    
    # 1: Catalog
    # 2: Pages
    # 3, 4, 5: Page objects
    # 6, 7, 8: Content stream objects
    # 9: Font object
    
    font_obj_id = 9
    page_ids = [3, 4, 5]
    content_ids = [6, 7, 8]

    # Catalog
    cat_obj = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    # Pages
    kids_str = " ".join([f"{pid} 0 R" for pid in page_ids])
    pages_obj = f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {len(page_ids)} >>\nendobj\n"
    
    page_objs = []
    content_objs = []

    for idx, (pid, cid, lines) in enumerate(zip(page_ids, content_ids, pages)):
        p_obj = (
            f"{pid} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {cid} 0 R "
            f"/Resources << /Font << /F1 {font_obj_id} 0 R >> >> >>\n"
            f"endobj\n"
        )
        page_objs.append(p_obj)

        # Build stream text
        stream_content = "BT\n/F1 11 Tf\n14.5 TL\n50 740 Td\n"
        for line in lines:
            if line == "":
                stream_content += "T*\n"
            else:
                stream_content += f"({escape_text(line)}) '\n"
        stream_content += "ET\n"
        stream_len = len(stream_content.encode('latin1'))

        c_obj = (
            f"{cid} 0 obj\n"
            f"<< /Length {stream_len} >>\n"
            f"stream\n"
            f"{stream_content}"
            f"endstream\n"
            f"endobj\n"
        )
        content_objs.append(c_obj)

    font_obj = (
        f"{font_obj_id} 0 obj\n"
        f"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        f"endobj\n"
    )

    all_objs = [cat_obj, pages_obj] + page_objs + content_objs + [font_obj]

    # Assemble PDF with cross reference table
    pdf_header = "%PDF-1.4\n"
    body = ""
    xref_offsets = [0]

    offset = len(pdf_header.encode('latin1'))
    for obj in all_objs:
        xref_offsets.append(offset)
        obj_bytes = obj.encode('latin1')
        offset += len(obj_bytes)
        body += obj

    xref_start = offset
    xref = f"xref\n0 {len(all_objs) + 1}\n"
    xref += "0000000000 65535 f \n"
    for off in xref_offsets[1:]:
        xref += f"{off:010d} 00000 n \n"

    trailer = (
        f"trailer\n"
        f"<< /Size {len(all_objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n"
        f"{xref_start}\n"
        f"%%EOF\n"
    )

    full_pdf_content = (pdf_header + body + xref + trailer).encode('latin1')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(full_pdf_content)
    print(f"Sample PDF created at: {output_path} ({len(full_pdf_content)} bytes)")

if __name__ == "__main__":
    out = "C:/Users/ANAND KOMMURI/.gemini/antigravity-ide/scratch/Intelligent-Legal-Document-Answering/documents/Sample_Service_and_NDA_Agreement.pdf"
    create_sample_pdf(out)
