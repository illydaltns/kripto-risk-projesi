CREATE OR REPLACE PROCEDURE p_ekleuye(
    p_ad TEXT, 
    p_soyad TEXT,
    p_cinsiyet TEXT,
    p_telefon TEXT,
    p_eposta TEXT,
    p_adres_id INT
)
LANGUAGE plpgsql
AS $$
DECLARE
BEGIN
    INSERT INTO uyeler(
        uyead,
        uyesoyad,
        cinsiyet,
        telefon,
        eposta,
        adres_id
    )
    VALUES(
        p_ad,
        p_soyad,
        p_cinsiyet,
        p_telefon,
        p_eposta,
        p_adres_id
    );

	RAISE NOTICE 'ekleme baiarılı';
END;
$$;

CALL p_ekleuye(
	'bilal', 
	'habeşi',
	'E',
	'05556667788',
	'bilal@example.com',
	'3'
	);




SELECT column_name
FROM information_schema.columns
WHERE table_name = 'uyeler';


SELECT * FROM uyeler
ORDER BY uye_id DESC
LIMIT 1;

select
	uyead as "Adı",
	upper(uyead) as "Büyük",
	lower(uyead) as "Küçük",
	substring(uyead from 1 for 2) as "İlk 2 karakter",
	replace(uyead, 'i' , '*') as "i -> *"
from
	uyeler
limit 10;


create or replace function F_TOPLA_SAYI(P1 INT, P2 INT)
RETURNS INTEGER
LANGUAGE PLPGSQL
AS $$
DECLARE
	TOPLAM INT;

BEGIN	
	TOPLAM:= P1 + P2;

RETURN TOPLAM;
END;
$$

SELECT*FROM F_TOPLA_SAYI(3,4)


create or replace function F_CARP_SAYI(P1 INT, P2 INT)
RETURNS INTEGER
LANGUAGE PLPGSQL
AS $$
DECLARE
	CARPIM INT;

BEGIN	
	CARPIM:= P1 * P2;

RETURN CARPIM;
END;
$$

SELECT*FROM F_CARP_SAYI(3,4)
