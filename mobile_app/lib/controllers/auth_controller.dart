import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:vegetable_classification/models/User.dart';
import 'package:vegetable_classification/views/auth/log_in_page.dart';
import 'package:vegetable_classification/views/home/home_page.dart';

class AuthController extends GetxController {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final GoogleSignIn _googleSignIn = GoogleSignIn();
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  Rxn<User> firebaseUser = Rxn<User>();
  late Rx<UserModel> currentUser;

  Future<void> loginWithGoogle() async {
    try {
      final GoogleSignInAccount? googleUser = await _googleSignIn.signIn();
      if (googleUser == null) return;



    } on FirebaseAuthException {
      Get.snackbar("Error", "Something went wrong");
    }
  }

  Future<void> login(String email, String password) async {
    try {
      await _auth.signInWithEmailAndPassword(email: email, password: password);
      DocumentSnapshot userDoc =
          await _firestore.collection("User").doc(_auth.currentUser!.uid).get();
      if (userDoc.exists) {
        currentUser.value = UserModel(
          name: userDoc['name'],
          email: userDoc['email'],
          createdAt: userDoc['createdAt'],
          uid: userDoc['uid'],
          avatarUrl: userDoc['avatarUrl'],
        );
        Get.off(HomePage());
      } else {
        print("User không tồn tại!");
      }
    } on FirebaseAuthException {
      Get.snackbar("Error", "Something went wrong");
    }
  }

  Future<void> register(String email, String password, String name) async {
    try {
      UserCredential userCredential = await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );
      UserModel user = UserModel(
        name: name,
        email: email,
        createdAt: DateTime.now(),
        uid: userCredential.user!.uid
      );
      await _firestore.collection("User").doc(userCredential.user!.uid).set(user.toJson());
      Get.off(LogInPage());
    } on FirebaseAuthException catch (e) {
      print(e.code.toString());
    }
  }
  void logOut () async {
    await _auth.signOut();
    firebaseUser.value = null;
    currentUser.value = UserModel(uid: "", name: "", email: "", createdAt: DateTime.now() , avatarUrl: "");
    Get.off(LogInPage());
  }

}
