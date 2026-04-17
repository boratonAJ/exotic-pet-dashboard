"""
Generate QR Code for the Exotic Pet Dashboard
This script creates a QR code that links to the deployed Streamlit dashboard
"""

import qrcode

# URL to the deployed dashboard
dashboard_url = "https://wwf-exotic-pet-dashboard.streamlit.app/"

# Create QR code instance
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

# Add data to the QR code
qr.add_data(dashboard_url)
qr.make(fit=True)

# Create an image from the QR code
img = qr.make_image(fill_color="black", back_color="white")

# Save the image
output_path = "dashboard_qr_code.png"
img.save(output_path)

print(f"✅ QR Code generated successfully!")
print(f"📁 Saved as: {output_path}")
print(f"🔗 URL: {dashboard_url}")
print(f"📏 Image size: {img.size}")
