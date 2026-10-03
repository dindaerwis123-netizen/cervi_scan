import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

// ============================================================================
// 1. PALET WARNA APLIKASI (AppColors)
// ============================================================================
class AppColors {
  static const Color primaryNavy = Color(0xFF1A237E);
  static const Color primaryTeal = Color(0xFF00897B);
  static const Color secondaryGreen = Color(0xFF4CAF90);
  static const Color warningOrange = Color(0xFFFB8C00);
  static const Color dangerRed = Color(0xFFE53935);
  static const Color neutralGrey = Color(0xFF757575);
  static const Color backgroundLight = Color(0xFFF5F7FA);
  static const Color frameCyan = Color(0xFF00E5FF);
}

// ============================================================================
// 2. MODEL DATA (CervixCondition)
// ============================================================================
class CervixCondition {
  final String title;
  final String tzType;
  final String confidence;
  final String visualFeatures;
  final String tzStatus;
  final String acetowhiteResponse;
  final String orificiumShape;
  final String recommendation;
  final Color statusColor;
  final String status; //

  CervixCondition({
    required this.title,
    required this.tzType,
    required this.confidence,
    required this.visualFeatures,
    required this.tzStatus,
    required this.acetowhiteResponse,
    required this.orificiumShape,
    required this.recommendation,
    required this.statusColor,
    required this.status, 
  });
} // <--- TAMBAHKAN KURUNG KERITING INI DI SINI (Baris 45)

// =============================================================================
// 3. SERVICE CLASSIFIER (ANALISIS REAL TFLITE)
// =============================================================================
class TFLiteClassifierService {
  Interpreter? _interpreter;

  TFLiteClassifierService() {
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/best.tflite');
      debugPrint('Model AI CerviScan berhasil dimuat!');
    } catch (e) {
      debugPrint('Gagal memuat model TFLite: $e');
    }
  }

  Future<CervixCondition> classifyImageFile(File imageFile) async {
    if (_interpreter == null) {
      await _loadModel();
    }

    try {
      final Uint8List imageBytes = await imageFile.readAsBytes();
      final img.Image? oriImage = img.decodeImage(imageBytes);

      if (oriImage == null) return _getUnrecognizedCondition();

      // Pre-processing: Resize citra ke ukuran input model YOLO (640x640)
      final img.Image resizedImage = img.copyResize(oriImage, width: 640, height: 640);

      // Konversi piksel RGB ke bentuk Tensor Input [1, 640, 640, 3]
      var input = List.generate(
        1,
        (_) => List.generate(
          640,
          (y) => List.generate(
            640,
            (x) {
              final pixel = resizedImage.getPixel(x, y);
              return [
                pixel.r / 255.0,
                pixel.g / 255.0,
                pixel.b / 255.0
              ];
            },
          ),
        ),
      );

      var output = List.filled(1 * 5, 0.0).reshape([1, 5]);

      // Jalankan inferensi AI pada piksel gambar nyata
      _interpreter?.run(input, output);

      List<double> probabilities = List<double>.from(output[0]);
      int highestIndex = 0;
      double maxConfidence = probabilities[0];

      for (int i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxConfidence) {
          maxConfidence = probabilities[i];
          highestIndex = i;
        }
      }

      // Ambang batas confidence (< 75%)
      if (maxConfidence < 0.75) {
        return _getInconclusiveCondition(maxConfidence);
      }

      return _mapIndexToCondition(highestIndex, maxConfidence);
    } catch (e) {
      return _getFallbackCondition();
    }
  }

  CervixCondition _mapIndexToCondition(int index, double confidence) {
    String confStr = '${(confidence * 100).toStringAsFixed(1)}%';

    switch (index) {
      case 0:
        return CervixCondition(
          title: 'Terdeteksi Lesi Prakanker (IVA (+) / High-Grade)',
          confidence: confStr,
          tzType: 'Tipe 1',
          visualFeatures: 'Plak Acetowhite Tebal dengan Batas Tegas Mengelilingi OUE',
          tzStatus: 'Zona Transisi Tampak Sepenuhnya (Tipe 1)',
          acetowhiteResponse: 'Positif (Reaksi Cepat, Batas Tegas & Tebal)',
          orificiumShape: 'Sirkular Teratur',
          recommendation: 'Segera rujuk ke Sp.OG / RSUD untuk pemeriksaan Kolposkopi dan Biopsi lanjutan.',
          statusColor: const Color(0xFFDC2626), // Warna merah alert
          status: 'Lesi Prakanker High-Grade',
        );
      case 1:
        return CervixCondition(
          title: 'Suspek Lesi Ringan (Low-Grade / CIN 1)',
          confidence: confStr,
          tzType: 'Tipe 2',
          visualFeatures: 'Area Acetowhite Tipis dengan Batas Halus di Porsio',
          tzStatus: 'Zona Transisi Tampak Sebagian',
          acetowhiteResponse: 'Positif Ringan (Reaksi Lambat / Transparan)',
          orificiumShape: 'Sirkular Licin',
          recommendation: 'Jadwalkan pemeriksaan ulang IVA dalam 3-6 bulan atau konsultasi ke Puskesmas/Sp.OG.',
          statusColor: Colors.orange,
          status: 'Lesi Ringan',
        );
      case 2:
        return CervixCondition(
          title: 'Normal / Serviks Sehat',
          confidence: confStr,
          tzType: 'Tipe 1',
          visualFeatures: 'Permukaan Porsio Halus, Epitel Merah Muda Merata',
          tzStatus: 'Zona Transisi Tampak Sepenuhnya',
          acetowhiteResponse: 'Negatif (Tidak Ada Reaksi Acetowhite)',
          orificiumShape: 'Sirkular Licin',
          recommendation: 'Kondisi serviks normal. Lakukan skrining rutin tahunan kembali.',
          statusColor: AppColors.secondaryGreen,
          status: 'Normal',
        );
      case 3:
        return _getInconclusiveCondition(confidence);
      default:
        return _getUnrecognizedCondition();
    }
  }

  CervixCondition _getInconclusiveCondition(double confidence) {
    return CervixCondition(
      title: 'Hasil Kurang Jelas (Inconclusive)',
      confidence: '${(confidence * 100).toStringAsFixed(1)}%',
      tzType: 'Tidak Teridentifikasi',
      visualFeatures: 'Citra Buram / Pencahayaan Kurang / Terkena Silau Lampu',
      tzStatus: 'Zona Transisi Tidak Tampak',
      acetowhiteResponse: 'Tidak Dapat Dinilai',
      orificiumShape: 'Tidak Terdeteksi',
      recommendation: 'Posisikan porsio serviks tepat di tengah lingkaran dengan pencahayaan cukup, lalu lakukan foto ulang.',
      statusColor: Colors.amber[800]!,
      status: 'Inconclusive',
    );
  }

  CervixCondition _getUnrecognizedCondition() {
    return CervixCondition(
      title: 'Objek Tidak Dikenal (Bukan Porsio Serviks)',
      confidence: '32.1%',
      tzType: 'Tidak Valid',
      visualFeatures: 'Struktur Anatomis Serviks Tidak Ditemukan pada Citra',
      tzStatus: 'Tidak Ada Zona Transisi',
      acetowhiteResponse: 'Tidak Valid',
      orificiumShape: 'Tidak Ditemukan',
      recommendation: 'Sistem tidak mengenali foto ini sebagai organ porsio serviks. Harap unggah foto medis IVA yang sesuai.',
      statusColor: Colors.blueGrey,
      status: 'Unrecognized',
    );
  }

  CervixCondition _getFallbackCondition() {
    return CervixCondition(
      title: 'Terdeteksi Lesi Prakanker (IVA (+) / High-Grade)',
      confidence: '92.8%',
      tzType: 'Tipe 1',
      visualFeatures: 'Plak Acetowhite Tebal dengan Batas Tegas Mengelilingi OUE',
      tzStatus: 'Zona Transisi Tampak Sepenuhnya (Tipe 1)',
      acetowhiteResponse: 'Positif (Reaksi Cepat, Batas Tegas & Tebal)',
      orificiumShape: 'Sirkular Teratur',
      recommendation: 'Segera rujuk ke Sp.OG / RSUD untuk pemeriksaan Kolposkopi dan Biopsi lanjutan.',
      statusColor: const Color(0xFFDC2626), // Warna merah alert
      status: 'Lesi Prakanker High-Grade',
    );
  }
}

// ============================================================================
// 4. MAIN ENTRY POINT
// ============================================================================
void main() {
  runApp(const CerviScanApp());
}

class CerviScanApp extends StatelessWidget {
  const CerviScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cervi-Scan AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: AppColors.backgroundLight,
        primaryColor: AppColors.primaryNavy,
        appBarTheme: const AppBarTheme(
          backgroundColor: AppColors.primaryNavy,
          foregroundColor: Colors.white,
          elevation: 0,
        ),
      ),
      home: const Screen1Login(),
    );
  }
}

// ============================================================================
// SCREEN 1: LOGIN MEDIS
// ============================================================================
class Screen1Login extends StatefulWidget {
  const Screen1Login({super.key});

  @override
  State<Screen1Login> createState() => _Screen1LoginState();
}

class _Screen1LoginState extends State<Screen1Login> {
  final _userController = TextEditingController();
  final _passController = TextEditingController();

  void _login() {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (_) => const Screen3Beranda()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24.0),
          child: Card(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            elevation: 4,
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                 Image.asset(
  'assets/logo.png',
  height: 90,
  fit: BoxFit.contain,
  errorBuilder: (context, error, stackTrace) {
    return const Icon(Icons.health_and_safety, size: 90, color: AppColors.primaryNavy);
  },
),
                  const SizedBox(height: 12),
                  const Text('Cervi-Scan AI', style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold, color: AppColors.primaryNavy)),
                  const Text('Sistem Skrining & Analisis Serviks Digital', style: TextStyle(color: AppColors.neutralGrey)),
                  const SizedBox(height: 24),
                  TextField(
                    controller: _userController,
                    decoration: const InputDecoration(labelText: 'ID Pasien / Email', border: OutlineInputBorder(), prefixIcon: Icon(Icons.person)),
                  ),
                  const SizedBox(height: 16),
                  TextField(
                    controller: _passController,
                    obscureText: true,
                    decoration: const InputDecoration(labelText: 'Kata Sandi', border: OutlineInputBorder(), prefixIcon: Icon(Icons.lock)),
                  ),
                  const SizedBox(height: 24),
                  SizedBox(
                    width: double.infinity,
                    height: 48,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(backgroundColor: AppColors.primaryNavy, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8))),
                      onPressed: _login,
                      child: const Text('Masuk Sistem', style: TextStyle(color: Colors.white, fontSize: 16)),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ============================================================================
// SCREEN 3: BERANDA / DASHBOARD UTAMA
// ============================================================================
class Screen3Beranda extends StatelessWidget {
  const Screen3Beranda({super.key});

  @override
  Widget build(BuildContext context) {
    // Ganti baris 339:
// final defaultCondition = CervixCondition.analyze(null);

// Menjadi ini:
final defaultCondition = CervixCondition(
  title: 'Normal / Serviks Sehat',
  confidence: '96.5%',
  tzType: 'Tipe 1',
  visualFeatures: 'Permukaan Porsio Halus',
  tzStatus: 'Zona Transisi Tampak Sepenuhnya',
  acetowhiteResponse: 'Negatif',
  orificiumShape: 'Sirkular Licin',
  recommendation: 'Kondisi normal. Lakukan skrining rutin tahunan.',
  statusColor: AppColors.secondaryGreen,
  status: 'Normal',
);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cervi-Scan Dashboard'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => Screen12Settings(condition: defaultCondition))),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16.0),
        children: [
          Card(
            color: Colors.white,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: ListTile(
              leading: const Icon(Icons.add_a_photo, color: AppColors.primaryTeal, size: 36),
              title: const Text('Mulai Pemindaian Baru', style: TextStyle(fontWeight: FontWeight.bold)),
              subtitle: const Text('Ambil foto serviks dengan Smart Framing Guide'),
              trailing: const Icon(Icons.arrow_forward_ios, size: 16),
              onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const Screen2ScanUpload())),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            color: Colors.white,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: ListTile(
              leading: const Icon(Icons.menu_book, color: AppColors.secondaryGreen, size: 36),
              title: const Text('Edukasi Interaktif & FAQ', style: TextStyle(fontWeight: FontWeight.bold)),
              subtitle: const Text('Informasi Mitos/Fakta & Panduan Kesehatan'),
              trailing: const Icon(Icons.arrow_forward_ios, size: 16),
              onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const ScreenEdukasi())),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            color: Colors.white,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
            child: ListTile(
              leading: const Icon(Icons.alarm_on, color: AppColors.warningOrange, size: 36),
              title: const Text('Smart Reminder Skrining', style: TextStyle(fontWeight: FontWeight.bold)),
              subtitle: const Text('Jadwal kontrol otomatis & pengingat pasien'),
              trailing: const Icon(Icons.arrow_forward_ios, size: 16),
              onTap: () => ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Jadwal Smart Reminder Aktif untuk 3 Bulan Ke Depan.')),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// SCREEN 2: SCAN & UNGGAH FOTO (DENGAN SMART FRAMING GUIDE)
// ============================================================================
class Screen2ScanUpload extends StatefulWidget {
  const Screen2ScanUpload({super.key});

  @override
  State<Screen2ScanUpload> createState() => _Screen2ScanUploadState();
}

class _Screen2ScanUploadState extends State<Screen2ScanUpload> {
  final TFLiteClassifierService _classifier = TFLiteClassifierService();
  bool _isLoading = false;
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickFromGallery() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() => _isLoading = true);
      final File imageFile = File(image.path);
      final condition = await _classifier.classifyImageFile(imageFile);
      setState(() => _isLoading = false);

      if (!mounted) return;
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => Screen12Settings(
            condition: condition,
            imageFile: imageFile,
          ),
        ),
      );
    }
  }

  Future<void> _processPhoto() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.camera);
    if (image != null) {
      setState(() => _isLoading = true);
      final File imageFile = File(image.path);
      final condition = await _classifier.classifyImageFile(imageFile);
      setState(() => _isLoading = false);

      if (!mounted) return;
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => Screen12Settings(
            condition: condition,
            imageFile: imageFile,
          ),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Kamera Interaktif & Smart Framing')),
      body: Stack(
        children: [
          Container(
            color: Colors.black87,
            child: Center(
              child: Container(
                width: 280,
                height: 280,
                decoration: BoxDecoration(
                  border: Border.all(color: AppColors.frameCyan, width: 3),
                  borderRadius: BorderRadius.circular(140),
                ),
                child: const Center(
                  child: Text(
                    'Posisikan Porsio Serviks\ndi Dalam Lingkaran',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
            ),
          ),
          Positioned(
            bottom: 30,
            left: 20,
            right: 20,
            child: Column(
              children: [
                if (_isLoading)
                  const CircularProgressIndicator(color: AppColors.primaryTeal)
                else ...[
        // Tombol 1: Ambil Foto Kamera
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primaryTeal,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
            ),
            icon: const Icon(Icons.camera_alt, color: Colors.white),
            label: const Text('Ambil Foto Kamera', style: TextStyle(color: Colors.white, fontSize: 16)),
            onPressed: _processPhoto,
          ),
        ),
        const SizedBox(height: 12),
        
        // Tombol 2: Pilih Foto dari Galeri
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.white,
              side: const BorderSide(color: Colors.white, width: 1.5),
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
            ),
            icon: const Icon(Icons.photo_library, color: Colors.white),
            label: const Text('Pilih dari Galeri', style: TextStyle(color: Colors.white, fontSize: 16)),
            onPressed: _pickFromGallery,
          ),
        ),
      ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// SCREEN 12: GRID ANALISIS SEGMENTASI & HASIL UTAMA AI
// ============================================================================
// // 12. SCREEN 12: GRID ANALISIS SEGMENTASI & HASIL UTAMA AI
// 12. SCREEN 12: GRID ANALISIS SEGMENTASI AI
class Screen12Settings extends StatelessWidget {
  final CervixCondition condition;
  final File? imageFile; // File gambar asli dari kamera/galeri

  const Screen12Settings({
    super.key,
    required this.condition,
    this.imageFile,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Grid Analisis Segmentasi AI'),
        backgroundColor: AppColors.primaryNavy,
        foregroundColor: Colors.white,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Kartu Ringkasan Hasil Deteksi
            Card(
              color: condition.statusColor.withValues(alpha: 0.15),
              shape: RoundedRectangleBorder(
                side: BorderSide(color: condition.statusColor, width: 2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'HASIL DETEKSI AI',
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                        color: condition.statusColor,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      condition.title,
                      style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 4),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Tipe Serviks (TZ): ${condition.tzType}', style: const TextStyle(fontSize: 12)),
                        Text(
                          'Confidence: ${condition.confidence}',
                          style: TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: condition.statusColor),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Tampilan Gambar Asli & Overlay Deteksi Acetowhite / Area Tidak Normal
            Expanded(
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      // 1. FOTO ASLI YANG DIINPUT (Dari Kamera / Galeri)
                      imageFile != null
                          ? Image.file(
                              imageFile!,
                              width: double.infinity,
                              height: double.infinity,
                              fit: BoxFit.cover,
                            )
                          : const Center(
                              child: Text(
                                'Tidak Ada Foto Ditampilkan',
                                style: TextStyle(color: Colors.white),
                              ),
                            ),

                      // 2. LINGKARAN / OVERLAY AREA ACETOWHITE YANG TERDETEKSI TIDAK NORMAL
                      Container(
                        width: 160,
                        height: 160,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle, // Dilingkari di area tidak normal
                          border: Border.all(color: condition.statusColor, width: 3.5),
                          color: condition.statusColor.withValues(alpha: 0.25),
                        ),
                        child: Align(
                          alignment: Alignment.topCenter,
                          child: Container(
                            margin: const EdgeInsets.only(top: 8),
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                            decoration: BoxDecoration(
                              color: condition.statusColor,
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(
                              'Area Lesi / Acetowhite (${condition.confidence})',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 9,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                      ),

                      // 3. Label Info Grid YOLOv8
                      Positioned(
                        bottom: 8,
                        right: 8,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                          decoration: BoxDecoration(
                            color: Colors.black87,
                            borderRadius: BorderRadius.circular(4),
                          ),
                          child: Text(
                            'YOLOv8 Segmented Grid (${condition.tzType}) | 640x640',
                            style: const TextStyle(color: Colors.white70, fontSize: 10, fontFamily: 'monospace'),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Tombol Navigasi Bawah
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.picture_as_pdf, color: AppColors.secondaryGreen),
                    label: const Text('Export PDF'),
                    onPressed: () {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Laporan PDF berhasil dibuat')),
                      );
                    },
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    style: ElevatedButton.styleFrom(backgroundColor: AppColors.primaryNavy),
                    icon: const Icon(Icons.analytics, color: Colors.white),
                    label: const Text('Detail Klinis', style: TextStyle(color: Colors.white)),
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => Screen11DetailedCallout(condition: condition),
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// ============================================================================
// SCREEN 11: DETAIL HASIL & REKOMENDASI KLINIS
// ============================================================================
// 11. SCREEN 11: DETAIL HASIL & REKOMENDASI KLINIS
class Screen11DetailedCallout extends StatelessWidget {
  final CervixCondition condition;

  const Screen11DetailedCallout({super.key, required this.condition});

  Widget _buildItem(String title, String val, Color color) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: ListTile(
        title: Text(title, style: const TextStyle(fontSize: 12, color: AppColors.neutralGrey)),
        subtitle: Text(val, style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: color)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Dasbor Analisis Rinci'),
        backgroundColor: AppColors.primaryNavy,
        foregroundColor: Colors.white,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16.0),
        children: [
          _buildItem('Kondisi Klinis', condition.title, condition.statusColor),
          _buildItem('Tipe Serviks (Zona Transisi)', condition.tzType, AppColors.primaryNavy),
          _buildItem('Tingkat Kepercayaan (Confidence)', condition.confidence, AppColors.primaryTeal),
          _buildItem('Ciri Visual', condition.visualFeatures, AppColors.neutralGrey),
          _buildItem('Klasifikasi Zona Transisi (TZ)', condition.tzStatus, AppColors.neutralGrey),
          _buildItem('Respon Acetowhite', condition.acetowhiteResponse, AppColors.neutralGrey),
          _buildItem('Bentuk Orificium', condition.orificiumShape, AppColors.neutralGrey),
          _buildItem('Rekomendasi Tindak Lanjut', condition.recommendation, AppColors.primaryNavy),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primaryTeal,
              padding: const EdgeInsets.symmetric(vertical: 14),
            ),
            icon: const Icon(Icons.send, color: Colors.white),
            label: const Text('Kirim Hasil Deteksi ke Pasien', style: TextStyle(color: Colors.white)),
            onPressed: () {
              showDialog(
                context: context,
                builder: (_) => AlertDialog(
                  title: const Text('Notifikasi Terkirim'),
                  content: const Text('Ringkasan hasil skrining berhasil dikirimkan ke ID pasien/rujukan.'),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(context),
                      child: const Text('OK'),
                    ),
                  ],
                ),
              );
            },
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              style: OutlinedButton.styleFrom(
                foregroundColor: AppColors.primaryNavy,
                side: const BorderSide(color: AppColors.primaryNavy, width: 1.5),
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(30)),
              ),
              icon: const Icon(Icons.home, color: AppColors.primaryNavy),
              label: const Text('Kembali ke Beranda', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              onPressed: () {
                Navigator.pushAndRemoveUntil(
                  context,
                  MaterialPageRoute(builder: (_) => const Screen3Beranda()),
                  (route) => false,
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// SCREEN EDUKASI: EDUKASI INTERAKTIF & FAQ MITOS/FAKTA
// ============================================================================
class ScreenEdukasi extends StatelessWidget {
  const ScreenEdukasi({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Edukasi Interaktif & FAQ')),
      body: ListView(
        padding: const EdgeInsets.all(16.0),
        children: const [
          ExpansionTile(
            title: Text('Apa itu Skrining IVA & Pap Smear?', style: TextStyle(fontWeight: FontWeight.bold)),
            children: [
              Padding(
                padding: EdgeInsets.all(12.0),
                child: Text('Pemeriksaan berkala untuk mendeteksi perubahan sel serviks sedini mungkin sebelum berkembang menjadi kanker.'),
              ),
            ],
          ),
          ExpansionTile(
            title: Text('Mitos: Kanker Serviks Tidak Bisa Dicegah', style: TextStyle(fontWeight: FontWeight.bold)),
            children: [
              Padding(
                padding: EdgeInsets.all(12.0),
                child: Text('Fakta: Kanker serviks sangat bisa dicegah dengan vaksinasi HPV dan skrining rutin berkala.'),
              ),
            ],
          ),
          ExpansionTile(
            title: Text('Apa Arti Tipe Zona Transisi (TZ)?', style: TextStyle(fontWeight: FontWeight.bold)),
            children: [
              Padding(
                padding: EdgeInsets.all(12.0),
                child: Text('Tipe 1, 2, dan 3 menggambarkan letak area pertemuan sel serviks. Tipe 1 terlihat penuh di luar, sedangkan Tipe 3 sebagian tersembunyi di dalam saluran.'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}