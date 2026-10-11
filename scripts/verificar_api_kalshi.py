from prediction_core import verify_public_catalog
if __name__ == "__main__":
    example=verify_public_catalog()
    print("API PÚBLICA KALSHI VERIFICADA")
    for key,value in example.items(): print(f"{key}: {value}")
