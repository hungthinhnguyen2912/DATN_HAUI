import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:mobile_app/components/input_text_field.dart';

import '../../../P.dart';

class EditProfilePage extends StatelessWidget {
  EditProfilePage({super.key});

  final TextEditingController _emailController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(),
      body: Column(
        children: [
          InputTextField(
            controller: _emailController,
            textWarning: "",
            hintText: P.auth.currentUser.value!.email,
            obs: false,
            readOnly: true,
          ),
        ],
      ),
    );
  }
}
