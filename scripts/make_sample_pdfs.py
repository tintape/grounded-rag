"""Generate synthetic demo PDFs for a fictional company, Lumora Home Robotics.

All facts are invented so the repo can be demoed and evaluated without any real or
confidential documents. Each fact lives on a known page, which makes citation checks exact.

Usage: python scripts/make_sample_pdfs.py [output_dir]
"""

import sys
from pathlib import Path

import pymupdf

PAGE_RECT = pymupdf.paper_rect("letter")
BODY_RECT = pymupdf.Rect(60, 70, PAGE_RECT.width - 60, PAGE_RECT.height - 60)
CSS = """
* { font-family: sans-serif; font-size: 11pt; line-height: 1.4; }
h1 { font-size: 18pt; color: #1f3a5f; }
h2 { font-size: 13pt; color: #1f3a5f; margin-top: 10pt; }
table { border-collapse: collapse; }
td, th { border: 1px solid #999; padding: 3pt 6pt; }
"""

DOCUMENTS: dict[str, tuple[str, list[str]]] = {
    "product-guide.pdf": (
        "Lumora S7 Robot Vacuum — Product Guide",
        [
            """<h1>Lumora S7 Robot Vacuum</h1><h2>What's in the box</h2>
            <p>The Lumora S7 ships with the robot, a charging dock, two side brushes, one HEPA
            filter (pre-installed) and a power cord.</p>
            <h2>Battery and runtime</h2>
            <p>The S7 uses a 5200 mAh lithium-ion battery. Runtime is up to 180 minutes in Eco
            mode, 120 minutes in Standard mode and 75 minutes in Max mode. A full charge from
            empty takes about 4 hours.</p>""",
            """<h2>Setup and app pairing</h2>
            <p>Install the Lumora Home app on iOS or Android and create an account. The S7 only
            supports 2.4 GHz Wi-Fi networks; 5 GHz networks are not supported.</p>
            <p>To enter pairing mode, hold the Power and Home buttons together for 3 seconds
            until the light ring pulses orange. Then follow the steps in the app.</p>""",
            """<h2>Cleaning modes and specifications</h2>
            <table><tr><th>Specification</th><th>Value</th></tr>
            <tr><td>Suction (Eco / Standard / Max)</td><td>1500 / 3000 / 5500 Pa</td></tr>
            <tr><td>Dustbin capacity</td><td>450 ml</td></tr>
            <tr><td>Noise level (Standard)</td><td>58 dB</td></tr>
            <tr><td>Weight</td><td>3.6 kg</td></tr></table>
            <p>Spot mode cleans a 1.5 m x 1.5 m square around the robot.</p>""",
            """<h2>Maintenance</h2>
            <p>Clean the main brush once a week. Replace the HEPA filter every 3 months and the
            side brushes every 6 months.</p>
            <h2>Factory reset</h2>
            <p>To restore factory settings, hold the Home and Spot buttons together for 10
            seconds until the light ring flashes blue. A factory reset erases saved maps and
            Wi-Fi settings.</p>""",
            """<h2>Safety</h2>
            <p>Do not use the S7 on wet floors or outdoors. The operating temperature range is
            0 to 40 °C. Keep the cliff sensors clean so the robot can detect stairs.</p>""",
        ],
    ),
    "warranty-and-returns.pdf": (
        "SOP-014 Warranty and Returns",
        [
            """<h1>SOP-014: Warranty and Returns</h1><h2>Return window</h2>
            <p>Customers may return any product within 30 days of delivery. Opened items
            returned without a defect are charged a 15% restocking fee. Defective items are
            returned with no fee.</p>""",
            """<h2>Warranty coverage</h2>
            <p>The robot is covered for 24 months from the purchase date. The battery is covered
            for 12 months. Brushes, filters and other consumables are not covered by the
            warranty. Proof of purchase is required for every claim.</p>""",
            """<h2>Agent procedure for a return (RMA)</h2>
            <ol><li>Verify the order number in OrderDesk.</li>
            <li>Collect the serial number. It is on the sticker under the dustbin and starts
            with LS7-.</li>
            <li>Create the RMA in OrderDesk.</li>
            <li>Email the customer a prepaid shipping label within 1 business day.</li></ol>
            <p>Refunds are issued within 5 business days after the return is received. Orders
            older than 60 days must be escalated to Tier 2 support.</p>""",
        ],
    ),
    "support-faq.pdf": (
        "Customer Support FAQ",
        [
            """<h1>Customer Support FAQ</h1><h2>Shipping</h2>
            <p>We ship to the United States and Canada only. Standard shipping takes 3 to 5
            business days and is free on orders over $199.</p>""",
            """<h2>Smart home and features</h2>
            <p>The S7 works with Amazon Alexa and Google Assistant. Apple HomeKit is not
            supported.</p>
            <p>The S7 does not mop. Mopping is available on the S7+ model, which includes a mop
            module. The app can store up to 4 floor maps for multi-level homes.</p>""",
            """<h2>Accounts and subscriptions</h2>
            <p>Customers can request deletion of their account data by emailing
            privacy@lumora.example; requests are completed within 30 days.</p>
            <p>The Lumora Care+ subscription costs $4.99 per month and extends the robot
            warranty to 36 months.</p>""",
        ],
    ),
    "troubleshooting.pdf": (
        "S7 Troubleshooting Guide",
        [
            """<h1>S7 Troubleshooting Guide</h1><h2>Error codes E1-E3</h2>
            <p><b>E1 — Wheel stuck.</b> Turn the robot over and remove hair or debris from both
            drive wheels.</p>
            <p><b>E2 — Cliff sensor blocked.</b> Wipe the four cliff sensors on the underside
            with a dry cloth.</p>
            <p><b>E3 — Side brush jammed.</b> Remove the side brush and clear tangled hair.</p>""",
            """<h2>Error codes E4-E7</h2>
            <p><b>E4 — Battery too low to return to dock.</b> Place the robot on the dock
            manually.</p>
            <p><b>E5 — Dustbin missing.</b> Reinsert the dustbin until it clicks.</p>
            <p><b>E7 — Wi-Fi connection failed.</b> Make sure the router broadcasts a 2.4 GHz
            network, restart the router, then pair the robot again.</p>""",
            """<h2>Robot is not charging</h2>
            <p>Clean the charging contacts on the robot and the dock with a dry cloth. The dock
            needs 0.5 m of clear space on each side and 1.5 m in front of it.</p>""",
        ],
    ),
}


def build_pdf(title: str, pages: list[str], path: Path) -> None:
    doc = pymupdf.open()
    total = len(pages)
    for number, html in enumerate(pages, start=1):
        page = doc.new_page(width=PAGE_RECT.width, height=PAGE_RECT.height)
        page.insert_text((60, 40), f"Lumora Home Robotics · {title}", fontsize=8, color=(0.4,) * 3)
        page.insert_htmlbox(BODY_RECT, html, css=CSS)
        footer = f"Page {number} of {total} · Synthetic demo document"
        page.insert_text((60, PAGE_RECT.height - 30), footer, fontsize=8, color=(0.4,) * 3)
    doc.set_metadata({"title": title, "author": "Lumora Home Robotics (fictional)"})
    doc.save(path, garbage=4, deflate=True)
    doc.close()


def main(output_dir: str = "data/sample_pdfs") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for file_name, (title, pages) in DOCUMENTS.items():
        build_pdf(title, pages, out / file_name)
        print(f"wrote {out / file_name} ({len(pages)} pages)")


if __name__ == "__main__":
    main(*sys.argv[1:])
