import math

# --- 1. Kullanıcıdan p ve q alma ---
p = int(input("Asal sayı p: "))
q = int(input("Asal sayı q: "))

# Modül hesaplama
n = p * q

# Euler Totient φ(n)
phi_n = (p - 1) * (q - 1)

print(f"\nModül (n) = {n}")
print(f"Euler Totient φ(n) = {phi_n}")

# --- 2. e seçimi (otomatik) ---
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# φ(n) ile aralarında asal olan ilk e'yi bulalım
e = 2
while e < phi_n:
    if gcd(e, phi_n) == 1:
        break
    e += 1

print(f"Açık anahtar üssü (e) = {e}")

# --- 3. d hesaplama (modüler ters) ---
def mod_inverse(e, phi):
    for d in range(1, phi):
        if (e * d) % phi == 1:
            return d
    return None

d = mod_inverse(e, phi_n)

print(f"Gizli anahtar üssü (d) = {d}")

print("-" * 40)

# --- 4. Kullanıcıdan mesaj alma ---
M = int(input("Gönderilecek mesaj (M < n): "))

# Şifreleme
C = pow(M, e, n)
print(f"\nŞifreli mesaj (C) = {C}")

# Çözme
M_cozum = pow(C, d, n)
print(f"Çözülmüş mesaj = {M_cozum}")
