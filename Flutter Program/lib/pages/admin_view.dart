import 'package:flutter/material.dart';
import 'package:stroketry3/pages/stroke_zip_home.dart';
import '../data_classes/paitient_summary.dart';
import '../supabase_functions/get_scan_records.dart';
import '../widgets/scan_row.dart';
import '../supabase_functions/is_user_admin.dart';

class AdminView extends StatefulWidget {
  const AdminView({super.key});

  @override
  State<AdminView> createState() => AdminViewState();
}

class AdminViewState extends State<AdminView> {
  final GetScanRecords _retrival = GetScanRecords();
  late Future<List<PatientSummary>> _future;
  bool _isAdmin = false; 

  @override
  void initState() {
    super.initState();
    _future = _retrival.fetchAllScansWithSlices();
    _loadAdminStatus(); 
  }

  Future<void> _loadAdminStatus() async {
  try {
    final adminStatus = await isAdmin();
    if (mounted) {
      setState(() {
        _isAdmin = adminStatus;
      });
    }
  } catch (e) {
    print('isAdmin error: $e');
    if (mounted) setState(() => _isAdmin = false);
  }
}

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Stroke Zip Classifier + Locator'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => setState(() {
              _future = _retrival.fetchAllScansWithSlices();
            }),
          ),
          if (_isAdmin) ...[ 
            FilledButton(
              onPressed: () => Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => const StrokeZipHome()),
              ),
              child: const Text("Home Page"),
            ),
            const SizedBox(width: 10),
          ],
        ],
      ),
      body: FutureBuilder<List<PatientSummary>>(
        future: _future,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(child: Text('ERROR : ${snapshot.error}'));
          }

          final scans = snapshot.data ?? [];

          if (scans.isEmpty) {
            return const Center(child: Text('No Scans Found'));
          }

          return ListView.separated(
            padding: const EdgeInsets.all(16),
            itemCount: scans.length,
            separatorBuilder: (_, __) => const SizedBox(height: 12),
            itemBuilder: (context, index) => ScanRow(scan: scans[index]),
          );
        },
      ),
    );
  }
}