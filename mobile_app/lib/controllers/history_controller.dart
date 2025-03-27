import 'package:get/get.dart';
import 'package:mobile_app/P.dart';
import 'package:mobile_app/models/History.dart';

class HistoryController extends GetxController {
  late Rx<List<History>> listHistory;
  Future<void> GetHistory() async {
    P.fireStore.collection("History").doc(P.authFirebase.currentUser!.uid).get();
  }
}