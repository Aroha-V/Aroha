import mongoose from 'mongoose';

const MessageSchema = new mongoose.Schema(
  {
    conversationId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Conversation',
      required: true,
      index: true,
    },
    role:    { type: String, enum: ['user', 'bot'], required: true },
    text:    { type: String, required: true },
    isError: { type: Boolean, default: false },
  },
  { timestamps: true }
);

export default mongoose.models.Message ||
  mongoose.model('Message', MessageSchema);
