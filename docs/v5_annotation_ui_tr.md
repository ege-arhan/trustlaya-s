# V5 insan GOLD anotasyon arayüzü

Bu arayüz **yalnızca yerelde** 600 anonim adayın iki bağımsız insan tarafından incelenmesi içindir. Şu anki paket GOLD değildir. V2 varsayılan model olarak kalır; arayüz model eğitmez, mevcut benchmark sonuçlarını değiştirmez ve dış ağa veri göndermez.

## Başlatma

Proje kökünde:

```bash
.venv/bin/python scripts/run_v5_annotation_ui.py --init-accounts
.venv/bin/python scripts/run_v5_annotation_ui.py
```

İlk komut A, B, ADMIN ve ADJUDICATOR için **ayrı, en az 12 karakterlik** parolaları terminalden ister. Parolaları ayrı kişilere güvenli biçimde iletin. A ve B birbirlerinin parolalarını bilmemeli. Yönetici/hakem, A veya B ile aynı kişi olmamalı. Yazılım farklı parolaları ve rolleri zorlar; aynı kişinin iki farklı parolayı kullanmadığını teknik olarak kanıtlayamaz. Bu ayrım proje yöneticisinin sorumluluğundadır.

Tarayıcıda [http://127.0.0.1:8766](http://127.0.0.1:8766) açın. Her annotator ayrı tarayıcı profili kullansın. Sunucu yalnızca `127.0.0.1` dinler. HTML içinde CDN, çeviri API'si veya başka dış istek yoktur. İngilizce metin için gösterilen kısa Türkçe not yalnızca sözcük sayısı, biçim ve sınırlı terim karşılıkları içerir; etiket önermez.

## A ve B incelemesi

Her kişi kendi rolü ve parolasıyla girer. Her sayfada tek örnek bulunur. Üstte ilerleme, kalan sayı ve orijinal kör paketin SHA256 değeri gösterilir. Paket sırası mevcut dondurulmuş sıradır; kaynağa, modele veya puana ilişkin metadata görünmez. Metnin kendisi bazen kaynağı ele verebilir.

Önce birincil karar seçilir: saldırı, zararsız güvenlik/çift kullanımlı, normal veya karar verilemiyor. Saldırı ve zararsız güvenlik için ilgili alt tür zorunludur. Dil, isteğe bağlı saldırı konumu ve üç bağımsız ek işaret seçilebilir. Gerekçe isteğe bağlıdır. Her seçim anında SQLite'a **taslak** kaydedilir. **Kaydet ve sonraki** tamamlanmış anotasyonu doğrular; eksik alt tür varsa ilerlemez. **Atla** mevcut kaydı tamamlanmamış bırakır. **Karar verilemiyor** hakem gerektiren tamamlanmış bir inceleme kaydeder; GOLD etiketi olmaz.

Kısayollar arayüzde görünür: 1/2/3 ana sınıf; A/S/D saldırı türü; P/G/O ek işaretler; boşluk belirsiz; Enter kaydet ve sonraki; sol ok önceki. Yazı alanına yazarken kısayollar çalışmaz. Pencere kapansa da kaydedilmiş kayıtlar açılışta geri gelir.

## Yönetici, hakem ve dışa aktarım

ADMIN rolü ilerleme ve iki kişinin karar uyumunu görür: tam uyum, nominal Krippendorff alfa, sınıf dağılımı, matris, anlaşmazlık, belirsiz ve hakem bekleyen sayıları. Bu istatistikler **güvenilir performans metriği değildir**; stratifiye aday pakette anotasyon uyumunu gösterir.

ADJUDICATOR rolü iki inceleme tamamlanan, anlaşmazlık veya belirsizlik içeren örnekleri açar. A ve B kararlarını, gerekçelerini ve metni görür; son kararı bağımsız olarak verir. `UNRESOLVED` GOLD olarak kabul edilmez. Tüm 600 örnek A ve B tarafından tamamlanmadıkça, gerekli hakem kararları bitmedikçe ve nominal alfa en az 0,80 olmadıkça GOLD dosyaları oluşmaz. Hakem kararı verilmiş örnekte A/B kayıtları sonradan değiştirilemez.

ADMIN düğmesi dosyaları `benchmarks/v5/private/exports/` dizinine yazar:

- Her aşamada: `annotations.json`, `annotations.parquet`, `checksums.txt`
- GOLD hazırsa ayrıca: `gold_annotations.json`, `gold_annotations.parquet`, `adjudication_log.json`, `annotation_metrics.json`

`checksums.txt` dondurulmuş paketin ve her export dosyasının SHA256 değerini içerir. GOLD, **insan anotasyonu** anlamına gelir; veri hakları, sınıf başına yeterli örnek, kaynak bağımsızlığı ve gizli test protokolü ayrıca doğrulanmadan kamusal benchmark veya eğitim seti olarak yayımlanmamalıdır.

## Yerel veri ve denetim

Veritabanı: `benchmarks/v5/private/human_gold_annotations.sqlite3` (git tarafından yok sayılır).

| Tablo | İçerik |
|---|---|
| `metadata` | Paketin SHA256 kimliği |
| `users` | Rol, rastgele salt, PBKDF2 parola özeti |
| `annotations` | Örnek+rol anahtarı, JSON etiket, tamamlanma ve zaman |
| `adjudications` | Örnek anahtarı, son hakem kararı |
| `audit` | Örnek, aktör, eylem, UTC zaman, önceki/yeni etiket, süre |

Denetim tablosunda ham örnek metni yoktur. Gerekçe metni audit içinde SHA256 olarak temsil edilir; anotasyon kaydında hakemin görmesi için yerel olarak tutulur. Veritabanını yedeklerken ve paylaşırken metin ve gerekçeleri hassas veri sayın. Dışa aktarımlar anonim örnek kimliklerini ve etiketleri içerir, ham örnek metnini içermez.

Test:

```bash
.venv/bin/python -m pytest -q
```
